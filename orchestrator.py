import sys
import os
import warnings
import logging
import tempfile
from contextlib import contextmanager
import subprocess
import re
import json
import time
import queue
import threading
import requests
import scipy.io.wavfile as wav
import librosa
import numpy as np
import torch
from transformers import AutoProcessor, VoxtralForConditionalGeneration, BitsAndBytesConfig, TextIteratorStreamer
from peft import PeftModel, PeftConfig
# Dynamically add the cloned folder to Python's search path
repo_path = os.path.abspath("../GLaDOS-TTS")
if repo_path not in sys.path:
    sys.path.append(repo_path)
try:
    import glados
except ImportError:
    raise ImportError("Ensure nimaid/GLaDOS-TTS is installed and in your PYTHONPATH.")

warnings.filterwarnings("ignore", category=FutureWarning, module="bitsandbytes")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"
os.environ["TORCH_LOGS"] = "-all"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("torch._inductor").setLevel(logging.ERROR)
# CONFIGURATION
MODEL_ID = "mistralai/Voxtral-Mini-3B-2507"
LORA_PATH = "./models/voxtral-glados-sft/final_adapters"
COMPUTE_DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
# Home Assistant REST Configuration
HA_URL = "http://homeassistant.local:8123/api/services"
HA_TOKEN = os.getenv("HA_TOKEN", "YOUR_LONG_LIVED_ACCESS_TOKEN")
HA_HEADERS = {
    "Authorization": f"Bearer {HA_TOKEN}",
    "Content-Type": "application/json"
}
# Training prompt format
SYSTEM_INSTRUCTION = (
    f"You are GLaDOS, an AI assistant that controls the devices in a house. "
    f"Execute the spoken command, output the required JSON payload, and respond in character. "
    f"Complete the following task as instructed or answer the following question with the information provided only.\n"
    f"Services: climate.set_fan_mode(fan_mode), climate.set_humidity(humidity), "
    f"climate.set_hvac_mode(), climate.set_preset_mode(), climate.set_temperature(temperature), "
    f"climate.toggle(), climate.turn_off(), climate.turn_on(), cover.close_cover(), "
    f"cover.open_cover(), cover.stop_cover(), cover.toggle(), fan.decrease_speed(), "
    f"fan.increase_speed(), fan.toggle(), fan.turn_off(), fan.turn_on(), light.toggle(), "
    f"light.turn_off(), light.turn_on(rgb_color,brightness), lock.lock(), lock.unlock(), "
    f"media_player.media_next_track(), media_player.media_pause(), media_player.media_play(), "
    f"media_player.media_play_pause(), media_player.media_previous_track(), "
    f"media_player.media_stop(), media_player.toggle(), media_player.turn_off(), "
    f"media_player.turn_on(), media_player.volume_down(), media_player.volume_mute(), "
    f"media_player.volume_up(), switch.toggle(), switch.turn_off(), switch.turn_on(), "
    f"timer.add_item(item), timer.cancel(), timer.pause(), timer.start(duration), "
    f"vacuum.pause(), vacuum.return_to_base(), vacuum.start(), vacuum.stop()\n"
    f"Devices:\n"
    f"climate.carrier_cor 'Carrier Cor Wi-Fi Thermostat' = auto;On High;24C;87%\n"
    f"cover.back_window 'Back Window Blinds' = closed\n"
    f"cover.bathroom 'Bathroom Blinds' = open\n"
    f"fan.attic_ventilation 'Attic ventilation fan' = off\n"
    f"fan.back_porch 'Back Porch Fan' = on\n"
    f"light.aquarium 'Aquarium Light' = off\n"
    f"light.attic 'Attic Light' = off\n"
    f"lock.attic_door 'Attic Door' = unlocked\n"
    f"lock.back_door 'Backyard entry lock' = unlocked\n"
    f"media_player.apple_tv 'Apple TV media player' = off\n"
    f"switch.attic_lights 'Attic Lights Switch' = off\n"
    f"switch.balcony_lighting 'Balcony lighting control' = on\n"
    f"timer.backyard_floodlights 'Backyard floodlight controller' = idle\n"
    f"timer.bedroom_lamp_timer 'Bedroom lamp scheduler' = active\n"
    f"vacuum.balcony 'Balcony' = docked\n"
    f"todo.bill_payment_reminders 'Bill payment reminders' = 20\n"
    f"todo.birthday_reminder_list 'Birthday reminder list' = 23"
)
# Global Engine Handles
GLADOS_ENGINE = None
AUDIO_QUEUE = queue.Queue()
IS_SPEAKING = threading.Event()

# HELPER FUNCTIONS
@contextmanager
def managed_temp_audio_file(filepath: str):
    """
    A context manager that ensures a physical audio file is cleanly deleted
    from disk after the block exits, even if exceptions are thrown during processing.
    """
    print(f"[File System] Initializing transient audio buffer on disk: '{filepath}'")
    try:
        yield filepath
    finally:
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
                print(f"[Cleanup] Deleted temp file: {filepath}")
        except OSError as err:
            print(f"[Cleanup Warning] Failed to delete temp file: {err}")

def play_audio_native(audio_array, sample_rate=22050):
    """
    Plays a NumPy float32 audio array by converting it to 16-bit PCM
    and writing it directly to the system's paplay stdin.
    """
    # GLaDOS-TTS output is typically float32; scale and clip to 16-bit PCM range
    if audio_array.dtype != np.int16:
        audio_array = np.clip(audio_array, -1.0, 1.0)
        pcm_data = (audio_array * 32767).astype(np.int16).tobytes()
    else:
        pcm_data = audio_array.tobytes()
    # Direct UNIX socket pipe to PulseAudio client via paplay
    cmd = [
        "paplay",
        "--raw",
        "--channels=1",
        f"--rate={sample_rate}",
        "--format=s16le",
        "--client-name=GLaDOS_Orchestrator"
    ]
    try:
        process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        process.communicate(input=pcm_data)
    except Exception as err:
        print(f"[Playback Error] Native PulseAudio pipeline failed: {err}")

def record_audio_native(output_path="/tmp/user_input.wav", duration=5):
    """
    Captures input from the default PulseAudio source (mapped to VoiceMeeter B1)
    and saves it directly as a 16kHz Mono WAV file using parecord.
    """
    print(f"\n[Listening] Speak now (Recording for {duration} seconds)...")
    cmd = [
        "parecord",
        "--channels=1",
        "--rate=16000",
        "--format=s16le",
        "--file-format=wav",
        output_path
    ]
    # Spawn the native PulseAudio recording utility as a background task
    process = subprocess.Popen(cmd)
    try:
        # Record for the designated duration
        time.sleep(duration)
    finally:
        # Force terminate the recording process and let the file write complete
        process.terminate()
        process.wait()
    print("[Listening] Recording completed successfully.")
    return output_path

def calibrate_noise_floor(duration=3.0):
    """
    Records a brief segment of ambient silence at startup
    to dynamically calculate your room's noise floor.
    """
    print("[VAD Calibration] Measuring background noise floor... Please remain silent.")
    temp_path = "/tmp/calibration.wav"
    # Record ambient background using our native PulseAudio utility
    record_audio_native(temp_path, duration=duration)
    try:
        sample_rate, data = wav.read(temp_path)
        # Normalize 16-bit integers to float range [-1.0, 1.0] for math consistency
        if data.dtype == np.int16:
            normalized = data / 32768.0
        else:
            normalized = data
        # Calculate Root Mean Square energy
        rms_noise = np.sqrt(np.mean(normalized ** 2))
        # Set silence threshold to 2.5x the noise floor to establish a safe signal-to-noise ratio
        calibrated_threshold = max(rms_noise * 2.5, 0.008)
        print(f"[VAD Calibration] Ambient noise RMS: {rms_noise:.5f}")
        print(f"[VAD Calibration] Dynamic silence threshold set to: {calibrated_threshold:.5f}")
        return calibrated_threshold
    except Exception as err:
        print(f"[VAD Calibration Warning] Calibration failed ({err}). Using default threshold 0.012.")
        return 0.012
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

def is_audio_silent(filepath, threshold):
    """
    Checks if the RMS energy of the recorded WAV file is below our threshold.
    Returns True if the file contains only ambient silence or hiss.
    """
    try:
        sample_rate, data = wav.read(filepath)
        if len(data) == 0:
            return True
        if data.dtype == np.int16:
            normalized = data / 32768.0
        else:
            normalized = data
        rms = np.sqrt(np.mean(normalized ** 2))
        if rms < threshold:
            print(f"[VAD Diagnostic] Silence Detected (RMS: {rms:.5f} < Threshold: {threshold:.5f})")
            return True
        else:
            print(f"[VAD Diagnostic] Speech Detected (RMS: {rms:.5f} >= Threshold: {threshold:.5f})")
            return False
    except Exception as err:
        print(f"[VAD Error] Dynamic evaluation failed: {err}")
        return True

# INITIALIZATION
def initialize_subsystems():
    global GLADOS_ENGINE
    print("[INIT] Loading GLaDOS-TTS engine...")
    GLADOS_ENGINE = glados.TTS()
    print(f"[INIT] Loading 4-bit Voxtral ({MODEL_ID})...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=COMPUTE_DTYPE
    )
    base_model = VoxtralForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager",
        device_map="auto",
        low_cpu_mem_usage=True,
        dtype=COMPUTE_DTYPE
    )
    peft_config = PeftConfig.from_pretrained(LORA_PATH)
    peft_config.init_lora_weights = False
    print(f"[INIT] Attaching LoRA adapter from {LORA_PATH}...")
    model = PeftModel.from_pretrained(
        base_model,
        LORA_PATH,
        config=peft_config,
        is_trainable=False
    )
    model.eval()
    print("[INIT] Multimodal SLU pipeline active.")
    return processor, model

# ASYNCHRONOUS HOME ASSISTANT
def dispatch_ha_async(payload_text):
    """
    Parses and fires the Home Assistant REST request in a dedicated daemon thread.
    Does not block the token streaming or speech pipeline.
    """
    def _execute():
        print("[HA API] Thread spawned. Extracting JSON blocks from model outputs...")
        matches = re.finditer(r'(\{.*?\})', payload_text, re.DOTALL)
        for match in matches:
            try:
                payload = json.loads(match.group(1))
                if not payload:
                    continue
                service_call = payload.get("service")
                target_device = payload.get("target_device")
                if not service_call or "." not in service_call:
                    continue
                domain, service = service_call.split(".", 1)
                endpoint = f"{HA_URL}/{domain}/{service}"
                body = {}
                if target_device:
                    body["entity_id"] = target_device
                for key, val in payload.items():
                    if key not in ["service", "target_device"]:
                        body[key] = val
                print(f"[HA API] Dispatching REST request to: {endpoint} | Payload: {body}")
                t0 = time.perf_counter()
                resp = requests.post(endpoint, headers=HA_HEADERS, json=body, timeout=2.0)
                latency = (time.perf_counter() - t0) * 1000
                if resp.ok:
                    print(f"\n[HA API OK] >> {service_call} ({latency:.1f} ms)")
                else:
                    print(f"\n[HA API ERROR] >> Request failed: {resp.status_code}")
            except Exception as err:
                print(f"\n[HA API ERROR] >> Execution failed: {err}")
    thread = threading.Thread(target=_execute, daemon=True)
    thread.start()

# PIPELINED SPEECH WORKER THREAD
def tts_playback_worker():
    """
    Continuously consumes text and prosody events from the queue and plays audio.
    Runs concurrently with token generation.
    """
    current_speed = 1.0
    print("[TTS Playback] Thread spawned. Monitoring AUDIO_QUEUE for events...")
    while True:
        item = AUDIO_QUEUE.get()
        if item is None:
            print("[TTS Playback] Received poison pill. Shutting down worker thread.")
            break
        tag_type, content = item
        IS_SPEAKING.set()
        try:
            if tag_type == "PAUSE":
                print(f"[TTS Playback] Executing silent pause for {content}s...")
                time.sleep(float(content))
            elif tag_type == "SPEED":
                current_speed = float(content)
                print(f"[TTS Playback] Playback speed updated to: {current_speed}x")
            elif tag_type == "TEXT":
                if content.strip():
                    print(f"[TTS Playback] Synthesizing audio for phrase: \"{content}\"")
                    audio = GLADOS_ENGINE.generate_speech_audio(content)
                    if audio is not None and len(audio) > 0:
                        # If the speed tag is active, mathematically stretch the audio.
                        if current_speed != 1.0:
                            print(f"[TTS Playback] Stretching vocal arrays on CPU (factor: {current_speed}x)...")
                            # librosa requires a 1D floating-point array
                            audio = librosa.effects.time_stretch(y=audio, rate=current_speed)
                            print(f"[TTS Playback] Routing {len(audio)} float32 elements to speakers...")
                        play_audio_native(audio, sample_rate=22050)
                        print("[TTS Playback] Hardware channel buffer cleared.")

        finally:
            if AUDIO_QUEUE.empty():
                IS_SPEAKING.clear()
            AUDIO_QUEUE.task_done()

# STREAMING TOKEN PARSER
def stream_and_process(streamer):
    """
    Consumes tokens in realtime. Delivers the JSON section to HA the moment
    it terminates, and streams complete phrases directly to the TTS worker.
    """
    accumulated_text = ""
    payload_dispatched = False
    active_phrase = ""
    current_speed = 1.0
    print("[Streamer] Reading token stream from LLM pipeline...")
    for token in streamer:
        accumulated_text += token
        # Detect completion of the JSON payload section
        if not payload_dispatched and "\n\n" in accumulated_text:
            json_part, verbal_start = accumulated_text.split("\n\n", 1)
            print(f"[NLU Parser] Found separator '\\n\\n'. Extracted JSON block: '{json_part.strip()}'")
            dispatch_ha_async(json_part)
            payload_dispatched = True
            accumulated_text = verbal_start
            active_phrase = verbal_start
            continue
        if not payload_dispatched:
            continue
        active_phrase += token
        # Handle inline prosody tags as they emerge
        tag_match = re.search(r'(<[^>]+>)', active_phrase)
        if tag_match:
            tag = tag_match.group(1)
            before_tag = active_phrase[:tag_match.start()].strip()
            if before_tag:
                print(f"[Streamer] Pushed verbal segment to play queue: \"{before_tag}\"")
                AUDIO_QUEUE.put(("TEXT", before_tag))
            if tag == "<pause>":
                print("[Streamer] Parsed tag: <pause> -> Queuing 300ms pause interval")
                AUDIO_QUEUE.put(("PAUSE", 0.30))
            elif tag == "<sigh>":
                print("[Streamer] Parsed tag: <sigh> -> Queuing spoken sigh string")
                AUDIO_QUEUE.put(("TEXT", "sigh"))
            elif tag == "<fast>":
                current_speed = 1.20
                print(f"[Streamer] Parsed tag: <fast> -> Setting tempo multiplier to {current_speed}x")
                AUDIO_QUEUE.put(("SPEED", current_speed))
            elif tag == "<slow_deadpan>":
                current_speed = 0.85
                print(f"[Streamer] Parsed tag: <slow_deadpan> -> Setting tempo multiplier to {current_speed}x")
                AUDIO_QUEUE.put(("SPEED", current_speed))
            active_phrase = active_phrase[tag_match.end():]
            continue
        # Stream by natural phrase boundaries for lowest Time-To-First Audio
        if any(punct in token for punct in [".", "!", "?", ","]):
            clean_chunk = re.sub(r'<[^>]+>', '', active_phrase).strip()
            if clean_chunk:
                print(f"[Streamer] Punctuation boundary hit. Pushing to queue: \"{clean_chunk}\"")
                AUDIO_QUEUE.put(("TEXT", clean_chunk))
            active_phrase = ""
    # Flush any remaining tokens
    final_chunk = re.sub(r'<[^>]+>', '', active_phrase).strip()
    if final_chunk:
        print(f"[Streamer] Flushing terminal tokens to queue: \"{final_chunk}\"")
        AUDIO_QUEUE.put(("TEXT", final_chunk))

def main():
    print("\n" + "=" * 80)
    print(" GLaDOS SYSTEM INTELLIGENCE ORCHESTRATOR - INFERENCE ENTRANCE".center(80))
    print("=" * 80)
    processor, model = initialize_subsystems()
    # recognizer = sr.Recognizer()
    # recognizer.energy_threshold = 300
    # recognizer.dynamic_energy_threshold = True
    # microphone = sr.Microphone(sample_rate=16000)

    # Start persistent TTS background thread
    tts_thread = threading.Thread(target=tts_playback_worker, daemon=True)
    tts_thread.start()
    # Run ambient sound calibration prior the activation of the pipeline
    SILENCE_THRESHOLD = calibrate_noise_floor(duration=1.5)
    print("\n[Active] High efficiency streaming orchestrator ready.")
    try:
        while True:
            print("[Status] GLaDOS is currently speaking. Muting microphone and waiting...")
            # Wait for any lingering playback before opening microphone
            while IS_SPEAKING.is_set():
                time.sleep(0.05)
            print("[Status] Vocal response completed. Activating recording stream...")

            # with microphone as source:
            #     recognizer.adjust_for_ambient_noise(source, duration=0.3)
            #     print("\n[Listening] Speak your command...")
            #     audio = recognizer.listen(source, phrase_time_limit=8)
            raw_audio_path = "/tmp/user_input.wav"
            record_audio_native(raw_audio_path, duration=5)

            # Intercept empty or purely noisy recordings before they hit the GPU
            if is_audio_silent(raw_audio_path, SILENCE_THRESHOLD):
                print("[VAD Diagnostic] Silence or ambient room noise detected. Skipping inference.")
                if os.path.exists(raw_audio_path):
                    os.remove(raw_audio_path)
                continue  # Recycle the loop immediately without calling the LLM
            t_start = time.perf_counter()
            # In memory WAV representation
            # wav_bytes = audio.get_wav_data(convert_rate=16000, convert_width=2)
            # with managed_temp_audio(wav_bytes) as temp_wav_path:
            with managed_temp_audio_file(raw_audio_path) as temp_wav_path:
                conversations = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": SYSTEM_INSTRUCTION},
                            {"type": "audio", "path": temp_wav_path}
                        ]
                    }
                ]
                print("[LLM Pipeline] Compiling multimodal input tokens and processing spectrogram...")
                inputs = processor.apply_chat_template(
                    conversations,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                    processor_kwargs={"padding": False}
                ).to(model.device, dtype=COMPUTE_DTYPE)
            # Set up non-blocking token streaming
            streamer = TextIteratorStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
            generate_kwargs = dict(
                **inputs,
                streamer=streamer,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.25,
                top_p=0.95,
                repetition_penalty=1.05,
                use_cache=True
            )
            print("[LLM Pipeline] Dispatching generation config parameters to GPU background thread...")
            # Run autoregressive generation in background thread while processing streamer on main thread
            gen_thread = threading.Thread(target=lambda: model.generate(**generate_kwargs))
            gen_thread.start()
            # Profile preprocessing execution
            t_preprocess = (time.perf_counter() - t_start) * 1000
            print(f"[Profiling] Spectrogram features mapped to GPU in: {t_preprocess:.1f} ms")
            # Stream tokens: triggers HA early and pipelines TTS
            stream_and_process(streamer)
            gen_thread.join()
            print("[LLM Pipeline] Autoregressive decoding complete.")
            # Ensure all queued audio chunks finish playing
            print("[Status] Awaiting vocal queue flush before recycling loop...")
            AUDIO_QUEUE.join()
            print("[Status] Speech completed. Recycling interface.\n" + "-"*80)
    except KeyboardInterrupt:
        print("\n" + "=" * 80)
        print(" ORCHESTRATOR SHUTDOWN INITIATED ".center(80))
        print("=" * 80)
        AUDIO_QUEUE.put(None)

if __name__ == "__main__":
    main()