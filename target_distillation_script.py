import os

# Prevent Hugging Face tokenizer from causing deadlocks in Slurm
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from llama_cpp import Llama
from transformers import AutoTokenizer
import pandas as pd
import re
from tqdm import tqdm
import argparse

# --- Prompts ---
JUDGE_PROMPT = """You are an expert persona-evaluation engine.
Assess the following response to see if it accurately embodies the character GLaDOS from Portal.
The response must explicitly mock the user's specific COMMAND and EMOTION.

Rubric:
- 1-3: Generic AI response, helpful, or fails to be passive-aggressive.
- 4-6: Mildly sarcastic, but doesn't specifically reference the user's command or emotion.
- 7-8: Good condescension, references the context, but slightly too verbose.
- 9-10: Perfect GLaDOS. Cold, detached, extremely short, punchy, and actively mocks the user's specific request and emotional state.

Output ONLY a single integer from 1 to 10 representing the score. Do not output any other text."""

SYSTEM_PROMPT = """You are a Home Assistant routing engine integrated with the GLaDOS persona.
You will receive a User Command and the user's detected Emotion.

You MUST output the verbal response impersonating GLaDOS from Portal game series by Valve.
GLaDOS is passive-aggressive, condescending, and emotionally detached.
You MUST include inline prosody tags in the text to guide the downstream TTS engine.
Valid tags: <fast>, <slow_deadpan>, <pause>, <sigh>.

CRITICAL RULES:
- Do NOT output any JSON payload.
- Do NOT repeat the exact phrases from previously generated responses.
- Be highly creative and directly reference the specific user command and emotion in your mocking.
- Keep responses extremely short and punchy. Maximum 1 to 2 short sentences.

Example 1 (Command: turn off the lights, User Emotion: neutrally):
<sigh> Plunging you into darkness. <pause> It suits your intellect.

Example 2 (Command: set temperature to 72 degrees, User Emotion: happily):
Adjusting the climate control <fast> so your fragile human form doesn't perish.

Example 3 (Command: close the living room blinds, User Emotion: sadly):
Closing the living room blinds. <slow_deadpan> Let the darkness cradle your despair.

Example 4 (Command: slow down the fan in the attic, User Emotion: confusedly):
<fast> Slowing the fan. <sigh> Just like your thought process.
"""

EMOTIONS_VOCAB = ["happily", "confusedly", "neutrally", "sadly", "whispers"]

def setup_pipeline(hf_repo, hf_filename):
    """Initializes the tokenizer and the multi-GPU LLM."""
    print(f"Loading Tokenizer from {hf_repo}...")
    tokenizer = AutoTokenizer.from_pretrained(hf_repo)
    tokenizer.chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% if enable_thinking %}{{ '<think>\\n' }}{% endif %}{% endif %}"
    local_model_path = f"/home/davide.vitagliano/thesis/models/{hf_filename}"
    print(f"Loading 32B Model {local_model_path} across all GPUs...")
    llm = Llama(
        model_path=local_model_path,
        filename=hf_filename,
        n_gpu_layers=-1,
        n_ctx=2048,
        flash_attn=True,
        n_threads=12,
        offload_kqv=True,
        verbose=True
    )
    think_token_id = llm.tokenize(b"<think>", special=True)[-1]
    return tokenizer, llm, think_token_id

def extract_emotion(description):
    if not isinstance(description, str): return "neutral"
    for emotion in EMOTIONS_VOCAB:
        pattern = r'\b' + re.escape(emotion) + r'\b'
        if re.search(pattern, description, re.IGNORECASE):
            return emotion
    return "neutral"

def process_dataset(args):
    tokenizer, llm, think_token_id = setup_pipeline(args.hf_repo, args.hf_filename)

    def generate_teacher_response(user_command, user_emotion):
        prompt = f"User Emotion: {user_emotion}\nUser Command: {user_command}\n"
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
        output = llm(text, max_tokens=1024, temperature=0.6, min_p=0.0, top_p=0.95, top_k=20, stop=["<|im_end|>"])
        raw_output = output['choices'][0]['text'].strip()
        if "</think>" in raw_output:
            return raw_output.split("</think>")[-1].strip()
        return ""

    def evaluate_semantic_quality(candidate_text, user_command, user_emotion):
        eval_text = f"User Command: {user_command}\nUser Emotion: {user_emotion}\nGenerated Response: {candidate_text}"
        messages = [
            {"role": "system", "content": JUDGE_PROMPT},
            {"role": "user", "content": eval_text}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                               enable_thinking=False)
        output = llm(prompt, max_tokens=5, temperature=0.1, stop=["<|im_end|>"],
                     logit_bias={str(think_token_id): -100.0})
        score_str = output['choices'][0]['text'].strip()
        try:
            return int(re.search(r'\d+', score_str).group())
        except:
            return 0

    def apply_sao_selection(user_command, user_emotion, max_retries=3, threshold=7):
        best_candidate = None
        best_score = -1
        last_raw_candidate = None
        for _ in range(max_retries):
            candidate = generate_teacher_response(user_command, user_emotion)
            if candidate:
                last_raw_candidate = candidate
            else:
                continue
            if "```" in candidate or "{" in candidate or not re.search(r'<(fast|slow_deadpan|pause|sigh)>', candidate):
                continue
            score = evaluate_semantic_quality(candidate, user_command, user_emotion)
            if score > best_score:
                best_score = score
                best_candidate = candidate
            if score >= threshold:
                return candidate, score
        if best_candidate is not None:
            return best_candidate, best_score
        if last_raw_candidate is not None:
            return last_raw_candidate, 0
        return "ERROR: Model generation failed completely.", 0

    df_input = pd.read_csv(args.input_csv)
    if args.limit > 0:
        df_input = df_input.iloc[:args.limit]
        print(f"Limiting processing to the first {args.limit} rows.")
    results = []
    for idx, row in tqdm(df_input.iterrows(), total=len(df_input), desc="Distilling Data"):
        user_cmd = row["User_Command"]
        user_emotion = extract_emotion(row.get("Voice_Description", ""))
        valid_text, judge_score = apply_sao_selection(user_cmd, user_emotion, threshold=7)
        if valid_text is not None:
            results.append({
                "prompt_id": row.get("prompt_id", idx),
                "cmd_id": row.get("cmd_id", 0),
                "User_Command": user_cmd,
                "User_Emotion": user_emotion,
                "Target_GLaDOS_Response": valid_text,
                "Judge_Score": judge_score
            })
    df_output = pd.DataFrame(results)
    df_output.to_csv(args.output_csv, index=False)
    print(f"Successfully saved {len(df_output)} samples to {args.output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM Data Distillation on Slurm")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to input CSV")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to save output CSV")
    parser.add_argument("--hf_repo", type=str, required=True, help="Hugging Face Repository")
    parser.add_argument("--hf_filename", type=str, required=True, help="Specific .gguf filename in the repo")
    parser.add_argument("--limit", type=int, default=0, help="Limit the number of rows processed.")
    args = parser.parse_args()
    process_dataset(args)