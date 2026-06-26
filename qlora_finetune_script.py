import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import (
    AutoProcessor,
    VoxtralForConditionalGeneration,
    BitsAndBytesConfig,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from transformers.trainer_utils import get_last_checkpoint
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
import os
import gc
import json
import wandb
from tqdm.auto import tqdm
import warnings


def create_datasets(df, commands_list, processor, test_size=0.1):
    def format_dataframe_to_dataset(frame, audio_dir="./data/synthesized_train_16k/"):
        dataset_dict = {"messages": []}
        missing_files = 0
        for _, row in frame.iterrows():
            full_audio_path = os.path.join(audio_dir, row["Audio_File"])
            # Validation check to prevent the trainer from crashing mid-epoch
            if not os.path.exists(full_audio_path):
                missing_files += 1
                print(f"Warning: Skipping {row['Audio_File']} - File not found at {full_audio_path}")
                continue
            target_output = f"{row['Assistant_Payload']}\n\n{row['Target_GLaDOS_Response']}{processor.tokenizer.eos_token}"
            conversation = [
                {"role": "user", "content": [{"type": "audio", "path": full_audio_path}]}, # The processor will handle loading and feature extraction during training
                {"role": "assistant", "content": [{"type": "text", "text": target_output}]},
                {"role": "user", "content": [{"type": "text", "text": "DUMMY_STOP"}]} # A dummy user turn to satisfy the Mistral Serving Validator
            ]
            dataset_dict["messages"].append(conversation)
        if missing_files > 0:
            print(f"Warning: Skipped {missing_files} rows due to missing audio files.")
        # No need to cast to datasets.Audio(), saving massive amounts of RAM
        return Dataset.from_dict(dataset_dict)

    train_cmds, eval_cmds = train_test_split(commands_list, test_size=test_size, random_state=42)
    # Filter the dataframe so all variations of a command stay strictly together
    df_train = df[df['User_Command'].isin(train_cmds)]
    df_eval = df[df['User_Command'].isin(eval_cmds)]
    print(f"Train rows: {len(df_train)} | Eval rows: {len(df_eval)}")
    train_ds = format_dataframe_to_dataset(df_train).shuffle(seed=42)
    eval_ds = format_dataframe_to_dataset(df_eval)
    return train_ds, eval_ds

def get_prepared_model(model_id, quantization_config, device, compute_dtype, processor):
    model = VoxtralForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        attn_implementation="sdpa",
        device_map=device
    )
    # Prepare model for gradient training
    model = prepare_model_for_kbit_training(model)
    # Essential for preventing backward pass crashes with frozen encoders
    model.enable_input_require_grads()
    # Revert the text embeddings back to bfloat16/float16 to match the Audio Encoder
    model.get_input_embeddings().to(compute_dtype)
    # Also ensure the output layer matches
    if getattr(model, "get_output_embeddings", None) is not None:
        model.get_output_embeddings().to(compute_dtype)
    # It is also good practice to ensure the audio encoder didn't get accidentally cast to float32
    if hasattr(model, "audio_encoder"):
        model.audio_encoder.to(compute_dtype)
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.eos_token_id
    model.config.bos_token_id = processor.tokenizer.bos_token_id
    return model

def make_voxtral_collate_fn(processor, chat_template):
    def voxtral_collate_fn(batch):
        # Re-apply dynamic attributes because PyTorch multiprocessing
        # drops them during Fast Tokenizer serialization.
        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
            processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
        if processor.tokenizer.chat_template is None:
            processor.tokenizer.chat_template = chat_template
        conversations = [item["messages"] for item in batch]
        inputs = processor.apply_chat_template(
            conversations,
            tokenize=True,
            return_dict=True,
            processor_kwargs={"padding": True, "return_tensors": "pt"}
        )
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        # Initialize labels for loss calculation
        labels = input_ids.clone()
        # Apply instruction masking logic
        inst_token_ids = processor.tokenizer.encode("[/INST]", add_special_tokens=False)
        seq_len = len(inst_token_ids)
        inst_seq = torch.tensor(inst_token_ids, device=labels.device)
        for i in range(labels.shape[0]):
            # Mask User Input and Left Padding via Sequence Matching
            # .unfold creates a sliding window of size `seq_len` to check for the exact token sequence
            matches = (labels[i].unfold(0, seq_len, 1) == inst_seq).all(dim=1)
            inst_indices = matches.nonzero(as_tuple=True)[0]
            if len(inst_indices) > 0:
                # Shift the index to the END of the [/INST] sequence
                first_inst_end_idx = inst_indices[0] + seq_len - 1
                # Mask everything up to and including the [/INST] sequence
                labels[i, :first_inst_end_idx + 1] = -100
            # Mask right-side dummy padding
            eos_indices = (labels[i] == processor.tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
            if len(eos_indices) > 0:
                # The first EOS belongs to our Assistant response
                target_eos_idx = eos_indices[0]
                # Safely mask everything after the target EOS token
                labels[i, target_eos_idx + 1:] = -100
                # Clean up attention mask and input ids for the dummy turn
                attention_mask[i, target_eos_idx + 1:] = 0
                input_ids[i, target_eos_idx + 1:] = processor.tokenizer.pad_token_id
        inputs["input_ids"] = input_ids
        inputs["attention_mask"] = attention_mask
        inputs["labels"] = labels
        return inputs
    return voxtral_collate_fn

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning, module="bitsandbytes")
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["WANDB_PROJECT"] = "Voxtral-GLaDOS-Multimodal"
    os.environ["WANDB_LOG_MODEL"] = "false"

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_main_process = (local_rank == 0)
    torch.cuda.set_device(local_rank)
    if is_main_process:
        wandb.login()

    model_id = "mistralai/Voxtral-Mini-3B-2507"
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    chat_template = (
        "{% for message in messages %}"
        "{% if message['role'] == 'user' %}[INST] "
        "{% if message['content'] is string %}{{ message['content'] }}"
        "{% else %}{% for block in message['content'] %}"
        "{% if block['type'] == 'text' %}{{ block['text'] }}"
        "{% elif block['type'] == 'audio' %}<audio>"
        "{% endif %}{% endfor %}{% endif %} [/INST]"
        "{% elif message['role'] == 'assistant' %}"
        "{% if message['content'] is string %}{{ message['content'] }}"
        "{% else %}{% for block in message['content'] %}"
        "{% if block['type'] == 'text' %}{{ block['text'] }}"
        "{% endif %}{% endfor %}{% endif %}"
        "{% endif %}{% endfor %}"
    )
    if is_main_process:
        print(f"Loading processor for {model_id}...")
    processor = AutoProcessor.from_pretrained(model_id)
    # Right padding for training
    processor.tokenizer.padding_side = "right"
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
    processor.tokenizer.chat_template = chat_template

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4", # Highly optimized for speed/accuracy
        bnb_4bit_use_double_quant=True, # Saves extra memory at no speed cost
        bnb_4bit_compute_dtype=compute_dtype
    )

    if is_main_process:
        print("Loading and preparing datasets...")
    df = pd.read_csv("./data/combined_multimodal_dataset_train.csv")
    # Extract all unique user commands
    unique_commands = df['User_Command'].unique().tolist()
    if is_main_process:
        print(f"Total Unique Commands: {len(unique_commands)}")

    best_params = {
        'learning_rate': 2e-4,
        'lora_r': 16,
        'lora_alpha': 32,
        'lora_dropout': 0.1
        }
    params_path = "./models/best_sweep_params.json"
    if os.path.exists(params_path):
        print(f"Found existing configuration at {params_path}")
        with open(params_path, "r") as f:
            best_params = json.load(f)
    train_dataset, eval_dataset = create_datasets(df, unique_commands, processor)
    del df, unique_commands
    gc.collect()
    torch.cuda.empty_cache()

    if is_main_process:
        print(f"Loading {model_id} in 4-bit precision...")
    output_dir = "./models/voxtral-glados-sft"
    run_id_file = os.path.join(output_dir, "wandb_run_id.txt")
    last_checkpoint = None
    wandb_run_id = None

    if os.path.exists(output_dir):
        last_checkpoint = get_last_checkpoint(output_dir)
        if last_checkpoint and os.path.exists(run_id_file):
            with open(run_id_file, "r") as f:
                wandb_run_id = f.read().strip()

    # Gate W&B Initialization to Main Process
    if is_main_process:
        if last_checkpoint and wandb_run_id:
            print(f"Resuming W&B run: {wandb_run_id}...")
            wandb.init(project="Voxtral-GLaDOS-Multimodal", id=wandb_run_id, resume="must")
        else:
            print("Starting a new W&B run...")
            wandb_run_id = wandb.util.generate_id()
            os.makedirs(output_dir, exist_ok=True)
            with open(run_id_file, "w") as f:
                f.write(wandb_run_id)
            wandb.init(project="Voxtral-GLaDOS-Multimodal", name="voxtral-GLaDOS", id=wandb_run_id, resume="allow")

    # 3. Strict Device Mapping for QLoRA
    model = get_prepared_model(
        model_id=model_id,
        quantization_config=bnb_config,
        device={"": local_rank},
        compute_dtype=compute_dtype,
        processor=processor
    )

    lora_config = LoraConfig(
        r=best_params['lora_r'],
        lora_alpha=best_params['lora_alpha'],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=best_params['lora_dropout'],
        bias="none",
        task_type="CAUSAL_LM"
    )

    training_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=4,
        max_length=1536,
        eval_strategy="steps",
        eval_steps=800,
        save_strategy="steps",
        save_steps=800,
        save_total_limit=3,
        load_best_model_at_end=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        gradient_accumulation_steps=8,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=2,
        learning_rate=best_params['learning_rate'],
        logging_steps=10,
        num_train_epochs=5,
        optim="paged_adamw_8bit",
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
        report_to="wandb",
        ddp_find_unused_parameters=False, # Essential for DDP + Gradient Checkpointing to prevent stalling/crashing
        neftune_noise_alpha=5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.01
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=make_voxtral_collate_fn(processor, chat_template),
        processing_class=processor,
        peft_config=lora_config
    )

    if is_main_process:
        trainer.model.print_trainable_parameters()
        print("Initiating QLoRA Multimodal Alignment...")
    try:
        if last_checkpoint is not None:
            if is_main_process:
                print(f"Resuming training from {last_checkpoint}...")
            trainer.train(resume_from_checkpoint=last_checkpoint)
        else:
            if is_main_process:
                print("Starting a new training run...")
            trainer.train()

        if is_main_process:
            trainer.save_model(os.path.join(output_dir, "final_adapters"))
            print("Training complete. Adapters saved.")
    finally:
        del model, trainer
        gc.collect()
        torch.cuda.empty_cache()
        if is_main_process and wandb.run is not None:
            wandb.finish()