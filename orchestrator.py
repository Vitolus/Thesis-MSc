import sys
import os
import warnings
import logging
import subprocess
import re
import json
import time
import queue
import threading
import requests
import difflib
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
TAG_REGEX = re.compile(r'(<[^>]+>)')
JSON_EXTRACT_REGEX = re.compile(r'(\{.*?\})', re.DOTALL)
# Training prompt format
SERVICES = [
    "cover.close_cover",
    "cover.open_cover",
    "cover.stop_cover",
    "fan.decrease_speed",
    "fan.increase_speed",
    "fan.turn_off",
    "fan.turn_on",
    "light.turn_off",
    "light.turn_on",
    "lock.lock",
    "lock.unlock",
    "media_player.media_next_track",
    "media_player.media_pause",
    "media_player.media_play",
    "media_player.media_previous_track",
    "media_player.media_stop",
    "media_player.turn_off",
    "media_player.turn_on",
    "media_player.volume_down",
    "media_player.volume_mute",
    "media_player.volume_up",
    "timer.cancel()",
    "timer.pause()",
    "timer.start(duration)"
]
DEVICES = [
    "cover.bathroom 'Bathroom Blinds' = open",
    "fan.attic_ventilation 'Attic fan' = off",
    "light.living_room 'Living Room Light' = off",
    "lock.back_door 'Backyard lock' = unlocked",
    "media_player.apple_tv 'Apple TV media player' = off",
    "timer.bedroom_lamp_timer 'Bedroom lamp scheduler' = active",
    "todo.birthday_reminder_list 'Birthday reminder list'"
]
SYSTEM_INSTRUCTION = (
    f"You are GLaDOS, an AI assistant that controls the devices in a house. "
    f"Execute the spoken command, output the required JSON payload, and respond in character. "
    f"Complete the following task as instructed or answer the following question with the information provided only.\n"
    f"Home configuration:\n"
    f"Services: {', '.join(SERVICES)}\n"
    f"Devices:\n"
    f"{'\n'.join(DEVICES)}"
)
# Strip parameters: "timer.start(duration)" -> "timer.start"
VALID_SERVICES = [srv.split("(")[0] for srv in SERVICES]
# Strip friendly names and states: "cover.bathroom 'Bathroom Blinds' = open" -> "cover.bathroom"
VALID_DEVICES = [dev.split(" ")[0] for dev in DEVICES]
# Global Engine Handles
GLADOS_ENGINE = None
AUDIO_QUEUE = queue.Queue()
IS_SPEAKING = threading.Event()

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
    print("[INIT] Multimodal SLU pipeline active.\n")
    return processor, model

# AUDIO FUNCTIONS
def play_audio(audio_array, sample_rate=22050):
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
        print(f"[Playback Error] Native PulseAudio pipeline failed: {err}\n")

def record_audio(output_path="./test/user_input.wav"):
    """
    Push-To-Talk recording flow.
    Launches parecord on a keypress trigger and terminates it on the second keypress.
    """
    input("[PTT] Press [ENTER] to START recording GLaDOS command...")
    print("[PTT] >>> RECORDING ACTIVE <<< Speak now...")
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
    # Wait for the user to trigger the stop
    input("[PTT] Press [ENTER] to STOP recording and analyze...")
    # Terminate the process and let it finalize WAV headers
    process.terminate()
    process.wait()
    print(f"[PTT] Recording stopped. Audio successfully saved to: '{output_path}'\n")
    return output_path

def calibrate_noise_floor(duration=3.0):
    """
    Records a brief segment of ambient silence at startup
    to dynamically calculate your room's noise floor.
    """
    temp_path = "/tmp/calibration.wav"
    # Record ambient background using our native PulseAudio utility
    cmd = [
        "parecord",
        "--channels=1",
        "--rate=16000",
        "--format=s16le",
        "--file-format=wav",
        temp_path
    ]
    process = subprocess.Popen(cmd)
    try:
        # Record for the designated duration
        time.sleep(duration)
    finally:
        # Force terminate the recording process and let the file write complete
        process.terminate()
        process.wait()
    try:
        sample_rate, data = wav.read(temp_path)
        # Normalize 16-bit integers to float range [-1.0, 1.0] for math consistency
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        # Calculate Root Mean Square energy
        rms_noise = np.sqrt(np.mean(data ** 2))
        # Set silence threshold to 2.5x the noise floor to establish a safe signal-to-noise ratio
        return max(rms_noise * 2.5, 0.008)
    except Exception as err:
        print(f"[VAD Calibration Warning] Calibration failed ({err}).\n")
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
            data = data.astype(np.float32) / 32768.0
        rms = np.sqrt(np.mean(data ** 2))
        return rms < threshold
    except Exception as err:
        print(f"[VAD Error] Silence evaluation failed: {err}")
        return True

# HOME ASSISTANT THREAD
def enforce_schema(query, choices, cutoff=0.75):
    """
    Fuzzy semantic matcher. Scores the LLM's output against the truth schema.
    Returns the closest valid match if confidence is above the cutoff, otherwise None.
    """
    if not query:
        return None
    matches = difflib.get_close_matches(query, choices, n=1, cutoff=cutoff)
    return matches[0] if matches else None


def dispatch_ha_async(payload_text):
    """
    Parses and fires the Home Assistant REST request in a dedicated daemon thread.
    Corrects and rewrites inputs before sending to prevent API errors.
    """
    def _execute():
        print("[HA API] Thread spawned. Extracting JSON blocks from model outputs.")
        matches = JSON_EXTRACT_REGEX.finditer(payload_text)
        for match in matches:
            try:
                payload = json.loads(match.group(1))
                if not payload:
                    continue
                raw_service = payload.get("service")
                raw_device = payload.get("target_device")
                # Validate and correct the service first
                safe_service = enforce_schema(raw_service, VALID_SERVICES, cutoff=0.65)
                if not safe_service:
                    print(f"[RAE BLOCKED] Invalid service hallucinated: '{raw_service}'")
                    continue
                # Perform Domain Constrained Sifting to isolate target devices
                target_domain = safe_service.split(".")[0] + "."
                domain_constrained_devices = [dev for dev in VALID_DEVICES if dev.startswith(target_domain)]
                # Match device against the constrained pool
                safe_device = enforce_schema(raw_device, domain_constrained_devices, cutoff=0.60)
                # Fallback to global pool if domain sifting yields no matches
                if not safe_device:
                    safe_device = enforce_schema(raw_device, VALID_DEVICES, cutoff=0.60)
                if not safe_device:
                    print(f"[RAE BLOCKED] Invalid target device hallucinated: '{raw_device}'")
                    continue
                # Map corrected values back to the execution payload
                domain, service = safe_service.split(".", 1)
                endpoint = f"{HA_URL}/{domain}/{service}"
                body = {"entity_id": safe_device}
                # Package any auxiliary parameters
                for key, val in payload.items():
                    if key not in ["service", "target_device"]:
                        body[key] = val
                print(f"[HA API] Dispatching REST request to: {endpoint} | Payload: {body}")
                t0 = time.perf_counter()
                resp = requests.post(endpoint, headers=HA_HEADERS, json=body, timeout=2.0)
                latency = (time.perf_counter() - t0) * 1000
                if resp.ok:
                    print(f"[HA API OK] >> {safe_service} ({latency:.1f} ms)\n")
                else:
                    print(f"[HA API ERROR] >> Request failed: {resp.status_code}\n")
            except Exception as err:
                print(f"[HA API ERROR] >> Execution failed: {err}\n")
    thread = threading.Thread(target=_execute, daemon=True)
    thread.start()

# SPEECH WORKER THREAD
def adjust_speech_cadence(audio_array, speed_mode, sample_rate=22050):
    """
    Uses a Phase Vocoder to stretch or compress audio in the frequency domain.
    This changes the speed while locking the pitch, avoiding mid word chopping.
    """
    # Skip DSP entirely for normal speech
    if speed_mode == 1.0:
        return audio_array
    # Ensure the matrix is flattened to 1D
    audio_array = np.asarray(audio_array).flatten()
    # Execute the Phase Vocoder algorithm
    # speed_mode > 1.0 makes it faster, speed_mode < 1.0 makes it slower
    stretched_audio = librosa.effects.time_stretch(y=audio_array, rate=speed_mode, n_fft=1024)
    max_amplitude = np.max(np.abs(stretched_audio))
    if max_amplitude > 1.0:
        stretched_audio = stretched_audio / max_amplitude
    return stretched_audio

def tts_playback_worker():
    """
    Continuously consumes text and prosody events from the queue and plays audio.
    Runs concurrently with token generation.
    """
    current_speed = 1.0
    print("[TTS Playback] Thread spawned. Monitoring AUDIO_QUEUE for events.\n")
    while True:
        item = AUDIO_QUEUE.get()
        if item is None:
            print("[TTS Playback] Received poison pill. Shutting down worker thread.\n")
            break
        tag_type, content = item
        IS_SPEAKING.set()
        try:
            if tag_type == "PAUSE":
                time.sleep(float(content))
            elif tag_type == "SPEED":
                current_speed = float(content)
            elif tag_type == "TEXT":
                if content.strip():
                    audio = GLADOS_ENGINE.generate_speech_audio(content)
                    if audio is not None and len(audio) > 0:
                        # If the speed tag is active, mathematically stretch the audio.
                        if current_speed != 1.0:
                            audio = adjust_speech_cadence(audio, current_speed, sample_rate=22050)
                        play_audio(audio, sample_rate=22050)
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
    payload_dispatched = False
    active_phrase = ""
    full_verbal_response = []
    for token in streamer:
        # Detect completion of the JSON payload section
        if not payload_dispatched:
            active_phrase += token
            if "\n\n" in active_phrase:
                json_part, verbal_start = active_phrase.split("\n\n", 1)
                print("=" * 60)
                print("[NLU Parser] EXTRAPOLATED JSON PAYLOAD:")
                print(json_part.strip())
                print("=" * 60)
                dispatch_ha_async(json_part)
                payload_dispatched = True
                clean_verbal_start = verbal_start.replace('.', '').replace('*', '')
                active_phrase = clean_verbal_start
                if clean_verbal_start:
                    full_verbal_response.append(clean_verbal_start)
            continue
        # Aggressively filter tokens before they hit the TTS buffer
        clean_token = token.replace('.', '').replace('*', '')
        if not clean_token:
            continue  # Skip processing if the token was just a dot/asterisk
        active_phrase += clean_token
        full_verbal_response.append(clean_token)
        # Handle inline prosody tags as they emerge
        match = TAG_REGEX.search(active_phrase)
        if match:
            tag = match.group(1)
            # Extract everything generated before the tag
            before_tag = active_phrase[:match.start()].strip()
            # Flush accumulated text to TTS BEFORE executing the tag's effect
            if before_tag:
                AUDIO_QUEUE.put(("TEXT", before_tag))
            # Dispatch the specific instruction
            if tag == "<pause>":
                AUDIO_QUEUE.put(("SPEED", 1.0))
                AUDIO_QUEUE.put(("PAUSE", 0.2))
            elif tag == "<sigh>":
                AUDIO_QUEUE.put(("SPEED", 1.0))
                AUDIO_QUEUE.put(("TEXT", "sigh"))
            elif tag == "<fast>":
                AUDIO_QUEUE.put(("SPEED", 1.15))
            elif tag == "<slow_deadpan>":
                AUDIO_QUEUE.put(("SPEED", 0.85))
            # Remove the processed portion from the buffer
            active_phrase = active_phrase[match.end():]
    # Stream Termination Flush
    final_chunk = TAG_REGEX.sub('', active_phrase).strip()
    if final_chunk:
        AUDIO_QUEUE.put(("TEXT", final_chunk))
    print("[LLM Pipeline] STREAM COMPLETE: GLaDOS Response Summary")
    print(f"Decoded Spoken Output: \"{''.join(full_verbal_response).strip()}\"")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    print("=" * 80)
    print(" GLaDOS SYSTEM INTELLIGENCE ORCHESTRATOR - INFERENCE ENTRANCE".center(80))
    print("=" * 80)
    processor, model = initialize_subsystems()
    # Start persistent TTS background thread
    tts_thread = threading.Thread(target=tts_playback_worker, daemon=True)
    tts_thread.start()
    # Run ambient sound calibration prior the activation of the pipeline
    print("[VAD Calibration] Measuring background noise floor... Please remain silent.")
    SILENCE_THRESHOLD = calibrate_noise_floor(duration=2.0)
    print(f"[VAD Calibration] Silence threshold set to: {SILENCE_THRESHOLD:.5f}")
    print("[Active] Orchestrator ready.")
    AUDIO_PATH = "./test/user_input.wav"
    try:
        while True:
            print("[Status] GLaDOS is currently speaking. Muting microphone and waiting...")
            # Wait for any lingering playback before opening microphone
            while IS_SPEAKING.is_set():
                time.sleep(0.05)
            print("[Status] Vocal response completed. Activating recording stream...\n")
            record_audio(AUDIO_PATH)
            # Intercept empty or purely noisy recordings before they hit the GPU
            if is_audio_silent(AUDIO_PATH, SILENCE_THRESHOLD):
                print("[VAD Diagnostic] Silence or ambient room noise detected. Skipping inference.\n")
                continue  # Recycle the loop immediately without calling the LLM
            t_start = time.perf_counter()
            conversations = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": SYSTEM_INSTRUCTION},
                        {"type": "audio", "path": AUDIO_PATH}
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
            print(f"[Profiling] Spectrogram features mapped to GPU in: {t_preprocess:.1f} ms\n")
            # Stream tokens: triggers HA early and pipelines TTS
            stream_and_process(streamer)
            gen_thread.join()
            print("[LLM Pipeline] Autoregressive decoding complete.")
            # Ensure all queued audio chunks finish playing
            print("[Status] Awaiting vocal queue flush before recycling loop...")
            AUDIO_QUEUE.join()
            print("[Status] Speech completed. Recycling interface.\n" + "-" * 80)
    except KeyboardInterrupt:
        print("\n" + "=" * 80)
        print(" ORCHESTRATOR SHUTDOWN INITIATED ".center(80))
        print("=" * 80)
        AUDIO_QUEUE.put(None)