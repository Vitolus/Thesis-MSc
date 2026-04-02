import torch
import torch.multiprocessing as mp
import os
import gc
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import soundfile as sf
from parler_tts import ParlerTTSForConditionalGeneration, ParlerTTSConfig
from transformers import AutoTokenizer, set_seed
from transformers.cache_utils import StaticCache


# --- GLOBAL CONFIGURATION ---
BATCH_SIZE = 10
MAX_LENGTH = 50
MAX_NEW_TOKENS = 1720
# Apply the patch globally
if not hasattr(StaticCache, "batch_size"):
    StaticCache.batch_size = BATCH_SIZE
    StaticCache.max_batch_size = BATCH_SIZE

# --- THE WORKER FUNCTION (Runs on each GPU) ---
def worker_process(gpu_id, df_chunk, output_dir):
    """This function is completely isolated. It loads its own model on its specific GPU."""
    device = f"cuda:{gpu_id}"
    print(f"[GPU {gpu_id}] Initializing process. Rows to process: {len(df_chunk)}")
    # 1. Enable TF32 for speed
    torch.set_float32_matmul_precision('high')
    torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True
    # 2. Load Model strictly to this specific GPU
    model_id = "parler-tts/parler-tts-mini-expresso"
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    config = ParlerTTSConfig.from_pretrained(model_id)
    config.decoder._attn_implementation = "flash_attention_2"
    model = ParlerTTSForConditionalGeneration.from_pretrained(model_id, config=config).to(device)
    # 3. Compile the Model
    compile_mode = "reduce-overhead"  # "default" or "reduce-overhead"
    model.generation_config.cache_implementation = "static"
    model.forward = torch.compile(model.forward, mode=compile_mode)
    # 4. Warmup Step
    print(f"[GPU {gpu_id}] Compiling (this takes a minute)...")
    inputs = tokenizer(BATCH_SIZE * ["this is for compilation"], return_tensors="pt", padding="max_length",
                       max_length=MAX_LENGTH).to(device)
    model_kwargs = {
        "input_ids": inputs.input_ids,
        "attention_mask": inputs.attention_mask,
        "prompt_input_ids": inputs.input_ids,
        "prompt_attention_mask": inputs.attention_mask,
    }
    n_steps = 1 if compile_mode == "default" else 2
    for _ in range(n_steps):
        _ = model.generate(**model_kwargs, max_new_tokens=MAX_NEW_TOKENS)
    print(f"[GPU {gpu_id}] Compilation complete! Starting generation.")
    # 5. The Generation Loop (Adapted from your batching script)
    set_seed(42)
    audio_names_dict = {}
    # Use tqdm only on GPU 0 to avoid messy console output
    iterator = range(0, len(df_chunk), BATCH_SIZE)
    if gpu_id == 0:
        iterator = tqdm(iterator, desc="Processing Batches")
    for i in iterator:
        batch_df = df_chunk.iloc[i: i + BATCH_SIZE]

        user_cmds = batch_df["User_Command"].tolist()
        descriptions = batch_df["Voice_Description"].tolist()
        prompt_ids = batch_df.index.tolist()
        cmd_ids = batch_df["cmd_id"].tolist()

        actual_batch_length = len(user_cmds)
        # Pad final batch if necessary
        if actual_batch_length < BATCH_SIZE:
            pad_amount = BATCH_SIZE - actual_batch_length
            user_cmds.extend(["dummy"] * pad_amount)
            descriptions.extend(["dummy"] * pad_amount)
        # Tokenize and Generate
        batch_inputs = tokenizer(descriptions, return_tensors="pt", padding="max_length", max_length=MAX_LENGTH).to(
            device)
        batch_prompts = tokenizer(user_cmds, return_tensors="pt", padding="max_length", max_length=MAX_LENGTH).to(
            device)
        with torch.inference_mode():  # Extra safety to prevent memory leaks
            generation = model.generate(
                input_ids=batch_inputs.input_ids,
                attention_mask=batch_inputs.attention_mask,
                prompt_input_ids=batch_prompts.input_ids,
                prompt_attention_mask=batch_prompts.attention_mask,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True, # Enable sampling for more natural variance
                temperature=1.05, # Increase temperature for more creative generation
                top_p=0.85, # Nucleus sampling to focus on top % of probability mass
                repetition_penalty=1.0, # Penalize repetition for more natural sounding text
                return_dict_in_generate=True
            )
        # Save audio
        for j in range(actual_batch_length):
            audio_len = generation.audios_length[j]
            audio_arr = generation.sequences[j, :audio_len].cpu().numpy().squeeze()
            prompt_id = prompt_ids[j]
            cmd_id = cmd_ids[j]
            filename = f"prompt_{prompt_id}_cmd_{cmd_id}_varia_{prompt_id % 3}.wav"
            filepath = os.path.join(output_dir, filename)
            sf.write(filepath, audio_arr, model.config.sampling_rate)
            audio_names_dict[prompt_id] = filename
    # Cleanup
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return audio_names_dict

# --- THE LAUNCHER (Run this on the main CPU thread) ---
def launch_multigpu_job(csv_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    # 1. Load the full dataset
    df = pd.read_csv(csv_path, index_col=0)
    # 2. Determine available GPUs
    num_gpus = torch.cuda.device_count()
    if num_gpus < 2:
        print("Warning: Only 1 GPU detected. Running sequentially.")
        num_gpus = 1
    # 3. Split the dataframe into equal chunks based on GPU count
    chunk_indices = np.array_split(range(len(df)), num_gpus)
    df_chunks = [df.iloc[indices] for indices in chunk_indices]
    # 4. Spawn processes
    print(f"Distributing dataset across {num_gpus} GPUs...")
    mp.set_start_method('spawn', force=True)  # Required for CUDA multiprocessing
    # Use a process pool to collect results easily
    with mp.Pool(num_gpus) as pool:
        # Submit tasks asynchronously
        results = [
            pool.apply_async(worker_process, args=(i, df_chunks[i], output_dir))
            for i in range(num_gpus)
        ]
        # Wait for all GPUs to finish and get their dictionaries
        all_dicts = [res.get() for res in results]
    # 5. Merge results back into the main dataframe
    merged_dict = {}
    for d in all_dicts:
        merged_dict.update(d)
    df["audio_file_name"] = df.index.map(merged_dict)
    updated_csv_path = csv_path.replace(".csv", "_with_audio.csv")
    df.to_csv(updated_csv_path)
    print(f"Finished Multiprocessing! Saved updated dataset to {updated_csv_path}")


if __name__ == '__main__':
    # 1. This is strictly required to initialize PyTorch multiprocessing safely
    mp.set_start_method('spawn', force=True)
    # 2. Set up command line arguments for the CLI
    parser = argparse.ArgumentParser(description="Multi-GPU Parler-TTS Batch Generation")
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to the input CSV file")
    parser.add_argument("-o", "--output", type=str, required=True, help="Directory to save the generated audio files")
    args = parser.parse_args()
    # 3. Launch the generation
    print(f"Starting script...")
    print(f"Input Data: {args.input}")
    print(f"Output Directory: {args.output}")
    launch_multigpu_job(args.input, args.output)