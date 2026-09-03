import sys
import os
import tempfile
from contextlib import contextmanager
import re
import json
import time
import queue
import threading
import requests
import speech_recognition as sr
import sounddevice as sd
import librosa
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
    "You are GLaDOS, an AI assistant that controls the devices in a house. "
    "Execute the spoken command, output the required JSON payload, and respond in character. "
    "Complete the following task as instructed or answer the following question with the information provided only.\n"
)
# Global Engine Handles
GLADOS_ENGINE = None
AUDIO_QUEUE = queue.Queue()
IS_SPEAKING = threading.Event()


@contextmanager
def managed_temp_audio(wav_bytes: bytes):
    """
    A context manager that guarantees the creation, safe closing,
    and absolute deletion of a temporary audio file on disk,
    even in the event of an unhandled runtime exception.
    """
    # Acquisition phase
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    try:
        temp_file.write(wav_bytes)
        # Close the write handle so other library processes can safely open it
        temp_file.close()

        # Hand over the absolute path to the 'with' scope
        yield temp_file.name

    finally:
        # Release/Cleanup phase
        try:
            if os.path.exists(temp_file.name):
                os.remove(temp_file.name)
        except OSError as err:
            # Handle OS level lock failures or file access violations gracefully
            print(f"[Cleanup Warning] Failed to remove transient file: {err}")

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
                t0 = time.perf_counter()
                resp = requests.post(endpoint, headers=HA_HEADERS, json=body, timeout=2.0)
                latency = (time.perf_counter() - t0) * 1000
                if resp.ok:
                    print(f"\n[HA OK] >> {service_call} ({latency:.1f} ms)")
                else:
                    print(f"\n[HA ERROR] >> Request failed: {resp.status_code}")
            except Exception as err:
                print(f"\n[HA ERROR] >> Execution failed: {err}")
    thread = threading.Thread(target=_execute, daemon=True)
    thread.start()

# PIPELINED SPEECH WORKER THREAD
def tts_playback_worker():
    """
    Continuously consumes text and prosody events from the queue and plays audio.
    Runs concurrently with token generation.
    """
    current_speed = 1.0
    while True:
        item = AUDIO_QUEUE.get()
        if item is None:
            break
        tag_type, content = item
        IS_SPEAKING.set()
        try:
            if tag_type == "PAUSE":
                time.sleep(0.35)
            elif tag_type == "SPEED":
                current_speed = float(content)
            elif tag_type == "TEXT":
                if content.strip():
                    audio = GLADOS_ENGINE.generate_speech_audio(content)
                    if audio is not None and len(audio) > 0:
                        # If the speed tag is active, mathematically stretch the audio.
                        if current_speed != 1.0:
                            # librosa requires a 1D floating-point array
                            audio = librosa.effects.time_stretch(y=audio, rate=current_speed)
                        sd.play(audio, samplerate=22050)
                        sd.wait()
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
    for token in streamer:
        accumulated_text += token
        # Detect completion of the JSON payload section
        if not payload_dispatched and "\n\n" in accumulated_text:
            json_part, verbal_start = accumulated_text.split("\n\n", 1)
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
                AUDIO_QUEUE.put(("TEXT", before_tag))
            if tag == "<pause>":
                AUDIO_QUEUE.put(("PAUSE", 0.30))
            elif tag == "<sigh>":
                AUDIO_QUEUE.put(("TEXT", "sigh"))
            elif tag == "<fast>":
                current_speed = 1.20
                AUDIO_QUEUE.put(("SPEED", current_speed))
            elif tag == "<slow_deadpan>":
                current_speed = 0.85
                AUDIO_QUEUE.put(("SPEED", current_speed))
            active_phrase = active_phrase[tag_match.end():]
            continue
        # Stream by natural phrase boundaries for lowest Time-To-First Audio
        if any(punct in token for punct in [".", "!", "?", ","]):
            clean_chunk = re.sub(r'<[^>]+>', '', active_phrase).strip()
            if clean_chunk:
                AUDIO_QUEUE.put(("TEXT", clean_chunk))
            active_phrase = ""
    # Flush any remaining tokens
    final_chunk = re.sub(r'<[^>]+>', '', active_phrase).strip()
    if final_chunk:
        AUDIO_QUEUE.put(("TEXT", final_chunk))

if __name__ == "__main__":
    processor, model = initialize_subsystems()
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 300
    recognizer.dynamic_energy_threshold = True
    microphone = sr.Microphone(sample_rate=16000)
    # Start persistent TTS background thread
    tts_thread = threading.Thread(target=tts_playback_worker, daemon=True)
    tts_thread.start()
    print("\n[Active] High-efficiency streaming orchestrator ready.")
    try:
        while True:
            # Wait for any lingering playback before opening microphone
            while IS_SPEAKING.is_set():
                time.sleep(0.02)
            with microphone as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.3)
                print("\n[Listening] Speak your command...")
                audio = recognizer.listen(source, phrase_time_limit=8)
            t_start = time.perf_counter()
            # In memory WAV representation
            wav_bytes = audio.get_wav_data(convert_rate=16000, convert_width=2)
            with managed_temp_audio(wav_bytes) as temp_wav_path:
                conversations = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": SYSTEM_INSTRUCTION},
                            {"type": "audio", "path": temp_wav_path}
                        ]
                    }
                ]
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
            # Run autoregressive generation in background thread while processing streamer on main thread
            gen_thread = threading.Thread(target=lambda: model.generate(**generate_kwargs))
            gen_thread.start()
            print(f"[Profiling] Preprocessing & prompt encoding took: {(time.perf_counter() - t_start) * 1000:.1f} ms")
            # Stream tokens: triggers HA early and pipelines TTS
            stream_and_process(streamer)
            gen_thread.join()
            # Ensure all queued audio chunks finish playing
            AUDIO_QUEUE.join()
    except KeyboardInterrupt:
        print("\n[Shutdown] Orchestrator halted.")
        AUDIO_QUEUE.put(None)
