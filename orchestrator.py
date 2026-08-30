import os
import json
import re
import tempfile
import torch
import requests
import speech_recognition as sr
from transformers import AutoProcessor, VoxtralForConditionalGeneration, BitsAndBytesConfig
from peft import PeftModel, PeftConfig

# Model Configuration
MODEL_ID = "mistralai/Voxtral-Mini-3B-2507"
LORA_PATH = "./models/voxtral-glados-sft/final_adapters"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Home Assistant Configuration
HA_URL = "http://homeassistant.local:8123/api/services"
HA_TOKEN = "YOUR_LONG_LIVED_ACCESS_TOKEN"
HA_HEADERS = {
    "Authorization": f"Bearer {HA_TOKEN}",
    "content-type": "application/json",
}
# The exact environment profile the model was trained on
HA_DATA = (
    "Services: climate.set_fan_mode(fan_mode), climate.set_humidity(humidity), "
    "climate.set_hvac_mode(), climate.set_preset_mode(), climate.set_temperature(temperature), "
    "climate.toggle(), climate.turn_off(), climate.turn_on(), cover.close_cover(), "
    "cover.open_cover(), cover.stop_cover(), cover.toggle(), fan.decrease_speed(), "
    "fan.increase_speed(), fan.toggle(), fan.turn_off(), fan.turn_on(), light.toggle(), "
    "light.turn_off(), light.turn_on(rgb_color,brightness), lock.lock(), lock.unlock(), "
    "media_player.media_next_track(), media_player.media_pause(), media_player.media_play(), "
    "media_player.media_play_pause(), media_player.media_previous_track(), media_player.media_stop(), "
    "media_player.toggle(), media_player.turn_off(), media_player.turn_on(), media_player.volume_down(), "
    "media_player.volume_mute(), media_player.volume_up(), switch.toggle(), switch.turn_off(), "
    "switch.turn_on(), timer.add_item(item), timer.cancel(), timer.pause(), timer.start(duration), "
    "vacuum.pause(), vacuum.return_to_base(), vacuum.start(), vacuum.stop()\n"
    "Devices:\n"
    "climate.carrier_cor 'Carrier Cor Wi-Fi Thermostat' = auto;On High;24C;87%\n"
    "climate.emerson_sensi_touch 'Emerson Sensi Touch Wi-Fi Thermostat' = heat;Auto High;16C;68%\n"
    "cover.basement 'Basement Blinds' = closed\n"
    "cover.bathroom 'Bathroom Blinds' = open\n"
    "fan.attic 'Attic' = off\n"
    "fan.back_porch 'Back Porch Fan' = on\n"
    "light.aquarium 'Aquarium Light' = off\n"
    "light.attic 'Attic Light' = off\n"
    "lock.attic 'Attic Door Lock' = unlocked\n"
    "lock.back_door 'Backyard entry lock' = unlocked\n"
    "media_player.apple_tv 'Apple TV media player' = off\n"
    "media_player.attic_vinyl_turntable 'Attic Vinyl Record Player' = off\n"
    "switch.attic_lights 'Attic Lights Switch' = off\n"
    "switch.balcony_lighting 'Balcony lighting control' = on\n"
    "timer.backyard_floodlights 'Backyard floodlight controller' = idle\n"
    "timer.bedroom_lamp_timer 'Bedroom lamp scheduler' = active\n"
    "vacuum.back_deck_sweeper 'Back deck area cleaner' = cleaning\n"
    "vacuum.balcony 'Balcony' = docked\n"
    "todo.bill_payment_reminders 'Bill payment reminders' = 20\n"
    "todo.birthday_reminder_list 'Birthday reminder list' = 23"
)
SYSTEM_INSTRUCTION = (
    f"You are GLaDOS, an AI assistant that controls the devices in a house. "
    f"Execute the spoken command, output the required JSON payload, and respond in character. "
    f"Complete the following task as instructed or answer the following question with the information provided only.\n{HA_DATA}"
)


# Model Initialization
def initialize_model():
    print("Loading Voxtral and QLoRA Adapters...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    processor.tokenizer.padding_side = "right"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    base_model = VoxtralForConditionalGeneration.from_pretrained(
        MODEL_ID,
        attn_implementation="flash_attention_2",
        device_map="auto",
        low_cpu_mem_usage=True,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16
    )
    peft_config = PeftConfig.from_pretrained(LORA_PATH)
    peft_config.init_lora_weights = False
    model = PeftModel.from_pretrained(base_model, LORA_PATH, config=peft_config, is_trainable=False)
    model.eval()
    print("Model loaded successfully.")
    return processor, model

# Core Functions
def listen_to_microphone():
    """Listens until the user stops speaking and returns a 16kHz WAV file path."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("\nAdjusting for ambient noise... Please wait.")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        print("Listening! Speak your command...")
        try:
            audio_data = recognizer.listen(source, timeout=5, phrase_time_limit=15)
            print("Audio captured. Processing...")
            # The model requires 16kHz. We enforce it here.
            wav_bytes = audio_data.get_wav_data(convert_rate=16000, convert_width=2)
            # Write to a temporary file for the processor to consume
            temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            temp_wav.write(wav_bytes)
            temp_wav.close()
            return temp_wav.name
        except sr.WaitTimeoutError:
            print("Timed out waiting for speech.")
            return None

def parse_output(text):
    """Splits JSON payloads from the textual response."""
    payloads = []
    parts = text.split('\n\n', 1)
    json_section = parts[0]
    text_response = parts[1] if len(parts) > 1 else "No text generated"
    matches = re.finditer(r'(\{.*?\})', json_section, re.DOTALL)
    for match in matches:
        try:
            payloads.append(json.loads(match.group(1)))
        except json.JSONDecodeError:
            pass
    return payloads, text_response

def execute_ha_payloads(payloads):
    """Parses JSON payloads and sends HTTP POST requests to Home Assistant."""
    for payload in payloads:
        if not payload:  # Skip empty {} requests
            continue
        try:
            domain, service = payload["service"].split(".")
            target_device = payload.get("target_device")
            # Construct API Endpoint
            url = f"{HA_URL}/{domain}/{service}" # There is no home configured so it will give code != 200
            # Construct JSON Body
            data = {"entity_id": target_device} if target_device else {}
            # Append any additional slots (like temperature, rgb_color, etc.)
            for key, value in payload.items():
                if key not in ["service", "target_device"]:
                    data[key] = value
            # Execute
            print(f"Executing HA Command: {domain}.{service} on {target_device}")
            response = requests.post(url, headers=HA_HEADERS, json=data)
            if response.status_code != 200:
                print(f"HA Error: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"Failed to parse or execute payload: {e}")

def speak_glados(text):
    """Sends the text to the GLaDOS TTS engine."""
    # Assuming we are hosting a local TTS server (like glados-tts).
    # If using a CLI tool, this could be replaced with a subprocess.run() call.
    print(f"\n[GLaDOS]: {text}")
    tts_url = "http://localhost:8124/synthesize"  # Adjust to your TTS server port
    try:
        # Example format for a standard TTS REST API
        requests.post(tts_url, json={"text": text})
    except requests.exceptions.ConnectionError:
        print("TTS Server not found. Outputting text only.")

if __name__ == "__main__":
    processor, model = initialize_model()
    while True:
        input("\nPress ENTER to activate microphone (or Ctrl+C to quit)")
        audio_path = listen_to_microphone()
        if not audio_path:
            continue
        conversation = [[
            {"role": "user", "content": [
                {"type": "text", "text": SYSTEM_INSTRUCTION},
                {"type": "audio", "path": audio_path}
            ]}
        ]]
        inputs = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        ).to(model.device, dtype=torch.bfloat16)

        print("Model generating response...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.25,
                top_p=0.95,
                repetition_penalty=1.05
            )
        input_len = inputs.input_ids.shape[1]
        gen_text = processor.batch_decode(outputs[:, input_len:], skip_special_tokens=True)[0]
        payloads, text_response = parse_output(gen_text)
        # Execute the physical actions
        if payloads:
            execute_ha_payloads(payloads)
        # Synthesize the voice
        if text_response:
            speak_glados(text_response)
        # Cleanup memory and temp files
        os.remove(audio_path)
        torch.cuda.empty_cache()