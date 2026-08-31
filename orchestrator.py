import sys
import os
import re
import json
import time
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
import requests
import speech_recognition as sr
import sounddevice as sd
import torch
from transformers import AutoProcessor, VoxtralForConditionalGeneration, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
# Dynamically add the cloned folder to Python's search path
repo_path = os.path.abspath("../GLaDOS-TTS")
if repo_path not in sys.path:
    sys.path.append(repo_path)
try:
    import glados
except ImportError:
    raise ImportError(
        "Could not find the 'glados' module. Make sure you cloned the repository "
        "into the same directory as this script."
    )

# CONFIGURATION AND FLAGS
# Set to False if lacking acoustic echo canceling hardware
ENABLE_FULL_DUPLEX = False
# Model and Adapter Paths
MODEL_ID = "mistralai/Voxtral-Mini-3B-2507"
LORA_PATH = "./models/voxtral-glados-sft/final_adapters"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
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
IS_SPEAKING = threading.Event()

# SUBSYSTEM INITIALIZATION
def initialize_glados_tts():
    """Initializes and prewarms the GLaDOS-TTS engine."""
    global GLADOS_ENGINE
    print("[TTS] Initializing GLaDOS-TTS engine...")
    GLADOS_ENGINE = glados.TTS()
    print("[TTS] GLaDOS-TTS operational.")

# Model Initialization
def initialize_model():
    """Loads 4-bit quantized Voxtral and injects finetuned GLaDOS LoRA adapters."""
    print(f"[Model] Loading Base Voxtral Processor and Model ({MODEL_ID})...")
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
    print(f"[Model] Injecting LoRA adapter weights from '{LORA_PATH}'...")
    model = PeftModel.from_pretrained(
        base_model,
        LORA_PATH,
        config=peft_config,
        is_trainable=False
    )
    model.eval()
    print("[Model] Multimodal SLU pipeline active.")
    return processor, model

# MODULAR AUDIO INPUT MANAGER
class AudioInputManager:
    """
    Encapsulates audio acquisition.
    Supports standard half duplex turn taking and optional full duplex with barge in interruption.
    """

    def __init__(self, full_duplex_enabled=False):
        self.full_duplex_enabled = full_duplex_enabled
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True
        self.microphone = sr.Microphone(sample_rate=16000)

    def capture_command(self):
        """
        Captures speech from the microphone.
        If half-duplex: ensures no output playback is currently active before listening.
        If full-duplex: monitors for barge-in and stops audio playback if user speaks.
        """
        global GLADOS_ENGINE

        # In half duplex mode, wait until any ongoing TTS playback completely finishes
        if not self.full_duplex_enabled:
            while IS_SPEAKING.is_set():
                time.sleep(0.05)
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=0.4)
            print("\n[Listening] Awaiting user command...")

            audio = self.recognizer.listen(source, phrase_time_limit=10)

            # Active only when full duplex flag is enabled
            if self.full_duplex_enabled and IS_SPEAKING.is_set():
                print("\n[Barge in] User speech detected during speech output. Halting TTS...")
                if GLADOS_ENGINE:
                    GLADOS_ENGINE.stop_audio()
                IS_SPEAKING.clear()

        # Save buffer to 16kHz mono WAV for Voxtral processing
        temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_wav.write(audio.get_wav_data(convert_rate=16000, convert_width=2))
        temp_wav.close()
        return temp_wav.name

# PARSING AND EXECUTION LOGIC
def parse_output(raw_text):
    """
    Splits Voxtral output into Home Assistant JSON payload and GLaDOS character speech.
    """
    parts = raw_text.split('\n\n', 1)
    json_section = parts
    verbal_response = parts if len(parts) > 1 else "No text generated"
    payloads = []
    matches = re.finditer(r'(\{.*?\})', json_section, re.DOTALL)
    for match in matches:
        try:
            parsed = json.loads(match.group(1))
            if parsed:  # Omit empty dictionary queries
                payloads.append(parsed)
        except json.JSONDecodeError:
            pass
    return payloads, verbal_response.strip()

def execute_ha_payloads(payloads):
    """Dispatches domotic service calls directly to Home Assistant REST API."""
    if not payloads:
        print("[Domotics] Informational query or empty payload. No device actions required.")
        return

    for payload in payloads:
        service_call = payload.get("service")
        target_device = payload.get("target_device")
        if not service_call or "." not in service_call:
            print(f"[Domotics Error] Invalid service format: '{service_call}'")
            continue
        domain, service = service_call.split(".", 1)
        endpoint = f"{HA_URL}/{domain}/{service}"
        body = {}
        if target_device:
            body["entity_id"] = target_device
        for key, val in payload.items():
            if key not in ["service", "target_device"]:
                body[key] = val
        try:
            response = requests.post(endpoint, headers=HA_HEADERS, json=body, timeout=3.0)
            if response.ok:
                print(f"[Domotics Success] >> Executed '{service_call}' on '{target_device}'")
            else:
                print(f"[Domotics Failed] >> Status {response.status_code}: {response.text}")
        except Exception as err:
            print(f"[Domotics Exception] >> Network error communicating with Home Assistant: {err}")

def speak_segment(tts_engine, text, speed=1.0):
    """
    Synthesizes and plays back speech for a segment at the specified speed.
    """
    if not text.strip():
        return

    # If pacing is standard, use standard engine playback
    if abs(speed - 1.0) < 1e-3:
        tts_engine.speak_text_aloud(text)
    else:
        # Generate raw audio array from nimaid GLaDOS-TTS
        # Output is typically a float32/int16 NumPy array at 22050 Hz
        audio = tts_engine.generate_speech_audio(text)
        if audio is not None and len(audio) > 0:
            # Base sample rate of the GLaDOS Tacotron/VITS vocoder
            base_sample_rate = getattr(tts_engine, "sample_rate", 22050)
            # Modulate playback sample rate to alter speed without buffer truncation
            adjusted_rate = int(base_sample_rate * speed)
            sd.play(audio, samplerate=adjusted_rate)
            sd.wait()

def speak_glados(text):
    """
    Parses temporal tags (<fast>, <slow_deadpan>, <pause>, <sigh>) and routes
    clean speech segments to nimaid GLaDOS-TTS with appropriate timing delays.
    """
    global GLADOS_ENGINE
    if not text or not GLADOS_ENGINE:
        return

    print(f"\n[GLaDOS Verbal Output]: {text}")
    IS_SPEAKING.set()
    current_speed = 1.0
    tokens = re.split(r'(<[^>]+>)', text)
    try:
        for token in tokens:
            token = token.strip()
            if not token:
                continue
            # Interruption check
            if not IS_SPEAKING.is_set():
                break
            # Syntactic pause delays
            if token == "<pause>":
                time.sleep(0.25)
            elif token == "<sigh>":
                speak_segment(GLADOS_ENGINE, "sigh", speed=current_speed)
            elif token == "<fast>":
                current_speed = 1.25
            elif token == "<slow_deadpan>":
                current_speed = 0.80
            else:
                clean_segment = re.sub(r'<[^>]+>', '', token).strip()
                if clean_segment:
                    speak_segment(GLADOS_ENGINE, clean_segment, speed=current_speed)
    finally:
        IS_SPEAKING.clear()

def dispatch_actions_in_parallel(payloads, verbal_response):
    """
    Dispatches Home Assistant service execution and GLaDOS speech synthesis
    concurrently using worker threads.
    """
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_ha = executor.submit(execute_ha_payloads, payloads)
        future_tts = executor.submit(speak_glados, verbal_response)
        # Wait for both asynchronous tasks to finalize before starting the next user turn
        future_ha.result()
        future_tts.result()

if __name__ == "__main__":
    initialize_glados_tts()
    processor, model = initialize_model()
    # Initialize the modular audio manager with our full-duplex flag
    audio_manager = AudioInputManager(full_duplex_enabled=ENABLE_FULL_DUPLEX)
    mode_label = "Full Duplex" if ENABLE_FULL_DUPLEX else "Half Duplex"
    print("\n" + "=" * 60)
    print(f" GLaDOS HOME ASSISTANT ORCHESTRATOR")
    print(f" Operating Mode: {mode_label}")
    print("=" * 60)
    try:
        while True:
            # Capture speech (synchronous in half duplex, interrupt aware in full duplex)
            audio_path = audio_manager.capture_command()
            # Structure multimodal chat context
            conversations = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": SYSTEM_INSTRUCTION},
                        {"type": "audio", "path": audio_path}
                    ]
                }
            ]
            try:
                inputs = processor.apply_chat_template(
                    conversations,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                    processor_kwargs={"padding": False}
                ).to(model.device, dtype=COMPUTE_DTYPE)
                # Model Inference
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=256,
                        do_sample=True,
                        temperature=0.25,
                        top_p=0.95,
                        repetition_penalty=1.05
                    )
                input_len = inputs.input_ids.shape
                generated_text = processor.decode(outputs[0, input_len:], skip_special_tokens=True)
                # Extract structured payloads and dialogue string
                payloads, verbal_response = parse_output(generated_text)
                # Parallel Execution: Dispatches HA network API calls and GLaDOS audio playback simultaneously
                dispatch_actions_in_parallel(payloads, verbal_response)
            except Exception as e:
                print(f"[Processing Error]: {e}")
            finally:
                if os.path.exists(audio_path):
                    os.remove(audio_path)
    except KeyboardInterrupt:
        print("\n[Shutdown] Terminating orchestrator runtime cleanly.")