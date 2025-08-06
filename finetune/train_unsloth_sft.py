# train.py

import os, json, math, random, time
from typing import Dict, Any, Tuple, List
from functools import partial

# --- Environment Toggles for Stability ---
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
os.environ.setdefault("TE_DISABLE", "1")
os.environ.setdefault("ACCELERATE_DISABLE_TE", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

# --- Imports ---
from unsloth import FastLanguageModel
from datasets import Dataset
from transformers import TrainingArguments, DataCollatorForLanguageModeling, AutoTokenizer, Trainer, default_data_collator
from peft import PeftModel
import torch
import numpy as np
import wandb
from data_prep import build_chat_dataset

# ==============================================================================
# 1. Configuration (driven by environment variables)
# ==============================================================================
BASE_MODEL = os.getenv("BASE_MODEL", "unsloth/gemma-3n-E4B-unsloth-bnb-4bit")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./unsloth_gemma3n_sft")
TRAIN_PATH = os.getenv("TRAIN_PATH", "mn_dataset_out_v6_trust/clean/train_chains.jsonl")
DEV_PATH   = os.getenv("DEV_PATH",   "mn_dataset_out_v6_trust/clean/dev_chains.jsonl")

# Training Hyperparameters
SEED = int(os.getenv("SEED", "42"))
MAX_SEQ_LEN = int(os.getenv("MAX_SEQ_LEN", "8192"))
PER_DEVICE_BS = int(os.getenv("BATCH_SIZE", "1"))
GRAD_ACCUM = int(os.getenv("GRAD_ACCUM", "16"))
LR = float(os.getenv("LR", "5e-6"))
EPOCHS = float(os.getenv("EPOCHS", "2.0"))
WD = float(os.getenv("WEIGHT_DECAY", "0.1"))
WARMUP_RATIO = float(os.getenv("WARMUP_RATIO", "0.03"))
BF16 = os.getenv("BF16", "true").lower() == "true"
USE_QLORA = os.getenv("USE_QLORA", "true").lower() == "true"
USE_GRAD_CHKPT = os.getenv("USE_GRAD_CHKPT", "true").lower() == "true"
MASK_PROMPT_LOSS = os.getenv("MASK_PROMPT_LOSS", "true").lower() == "true"

# LoRA Configuration
LORA_R = int(os.getenv("LORA_R", "16"))
LORA_ALPHA = int(os.getenv("LORA_ALPHA", "32"))
LORA_DROPOUT = float(os.getenv("LORA_DROPOUT", "0.05"))
LORA_TARGETS = ""#["q_proj","k_proj","v_proj","o_proj"] #os.getenv("LORA_TARGETS", "q_proj,k_proj,v_proj,o_proj").strip()

# Logging & Saving
LOGGING_STEPS = int(os.getenv("LOGGING_STEPS", "20"))
SAVE_TOTAL_LIMIT = int(os.getenv("SAVE_TOTAL_LIMIT", "3"))
REPORT_TO = os.getenv("REPORT_TO", "wandb")

# ==============================================================================
# 2. Reproducibility & Data Formatting
# ==============================================================================
def seed_all(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def build_texts(example: Dict[str, Any]) -> Tuple[str, str]:
    """
    Builds a prompt-target pair from a multi-turn conversation.
    The prompt is all messages EXCEPT the final assistant message.
    The target is the content of ONLY the final assistant message.
    """
    messages = example.get("messages", [])
    if not messages or len(messages) < 2:
        return "", ""

    last_asst_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "assistant":
            last_asst_idx = i
            break
    
    if last_asst_idx == -1: return "", ""

    prompt_messages = messages[:last_asst_idx]
    target_message = messages[last_asst_idx]

    prompt_str = "".join([f"<|{msg['role']}|>\n{msg['content']}\n" for msg in prompt_messages])
    prompt_str += "<|assistant|>\n"
    target_str = target_message.get("content", "")

    return prompt_str, target_str

def example_is_valid(example: Dict[str, Any]) -> bool:
    """Quickly filters out examples that won't produce a valid prompt/target."""
    prompt, target = build_texts(example)
    return bool(prompt and target)

def tokenize_mask(example: Dict[str, Any], tokenizer, max_len: int, mask_prompt: bool) -> Dict[str, Any]:
    """Tokenizes prompt+target, and creates labels with loss masking on the prompt."""
    prompt, target = build_texts(example)
    if not prompt or not target:
        return {"input_ids": [], "labels": []}

    # Tokenize the full conversation string (prompt + target)
    # Using `add_special_tokens=False` can prevent extra BOS tokens if the prompt format already includes them.
    full_text = prompt + target + tokenizer.eos_token
    toks_full = tokenizer(full_text, truncation=True, max_length=max_len, add_special_tokens=False)
    
    # Tokenize the prompt-only part to find its length
    toks_prompt = tokenizer(prompt, truncation=True, max_length=max_len, add_special_tokens=False)

    input_ids = toks_full["input_ids"]
    labels = list(input_ids)
    
    if mask_prompt:
        prompt_len = len(toks_prompt["input_ids"])
        for i in range(min(prompt_len, len(labels))):
            labels[i] = -100 # -100 is the ignore_index for cross-entropy loss

    return {"input_ids": input_ids, "labels": labels}

class CustomDataCollator:
    """
    Custom data collator that manually pads the sequences and handles language modeling tasks.
    This is a workaround for models whose tokenizers lack a .pad() method.
    """
    def __init__(self, tokenizer, mlm=False):
        self.tokenizer = tokenizer
        self.mlm = mlm # Although mlm=False, we keep it for consistency
    
    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        tokenizer = self.tokenizer
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = tokenizer.eos_token_id

        # Find the max length in the batch for padding
        max_length = max(len(e["input_ids"]) for e in examples)

        # Manually pad each example in the batch
        for ex in examples:
            ids = ex["input_ids"]
            labs = ex["labels"]
            
            pad_len = max_length - len(ids)
            
            ex["input_ids"] = ids + ([pad_token_id] * pad_len)
            ex["labels"] = labs + ([-100] * pad_len)
            ex["attention_mask"] = [1] * len(ids) + [0] * pad_len

        # Use the default_data_collator to convert the list of dicts to a dict of tensors
        # This is the safe, standard way to do this final step.
        batch = default_data_collator(examples)

        # The default collator doesn't handle MLM, so if we were doing that,
        # we would do it here. Since mlm=False, we are done.
        return batch

# ==============================================================================
# 3. Main Training Orchestration
# ==============================================================================
def main():

    def get_local_rank():
        lr = os.environ.get("LOCAL_RANK")
        return int(lr) if lr is not None else 0

    def set_device_from_local_rank():
        local_rank = get_local_rank()
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            print(f"[PID {os.getpid()}] RANK={os.environ.get('RANK')} LOCAL_RANK={local_rank} "
                  f"WORLD_SIZE={os.environ.get('WORLD_SIZE')} -> cuda:{local_rank}")
    
    # In main() as the first thing after seeding:
    set_device_from_local_rank()
    seed_all(SEED)

    # --- Data Loading and Preparation ---
    train_chats, dev_chats = build_chat_dataset(TRAIN_PATH, DEV_PATH, system_prompt_path="distilled_prompt.txt")

    
    train_ds = Dataset.from_list(train_chats).filter(example_is_valid, num_proc=8)
    eval_ds = Dataset.from_list(dev_chats).filter(example_is_valid, num_proc=8)
    
    print(f"Train conversations after filter: {len(train_ds)}")
    print(f"Eval conversations after filter: {len(eval_ds)}")

    # --- Model and Tokenizer Loading ---
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL, max_seq_length=MAX_SEQ_LEN, load_in_4bit=USE_QLORA,
        dtype=None, device_map=None, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # CRITICAL FIX: Load a separate, standard tokenizer for the `.map()` function.
    # This avoids issues with Unsloth's patched tokenizer during multiprocessing.
    print("Loading a separate, standard tokenizer for data mapping...")
    map_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if map_tokenizer.pad_token is None:
        map_tokenizer.pad_token = map_tokenizer.eos_token

    # --- PEFT (LoRA) Configuration ---
    model = FastLanguageModel.get_peft_model(
        model, r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT, bias="none",
        target_modules=LORA_TARGETS.split(',') if LORA_TARGETS else "all-linear"
    )

    # --- Tokenization ---
    tok_map_fn = partial(tokenize_mask, tokenizer=map_tokenizer, max_len=MAX_SEQ_LEN, mask_prompt=MASK_PROMPT_LOSS)
    num_proc = max(1, min(8, os.cpu_count() or 1))
    
    train_tok = train_ds.map(tok_map_fn, remove_columns=train_ds.column_names, num_proc=num_proc)
    eval_tok = eval_ds.map(tok_map_fn, remove_columns=eval_ds.column_names, num_proc=num_proc)

    # --- Trainer Setup ---
    steps_per_epoch = math.ceil(len(train_tok) / (PER_DEVICE_BS * GRAD_ACCUM))
    
    args = TrainingArguments(
        per_device_train_batch_size=PER_DEVICE_BS,
        gradient_accumulation_steps=GRAD_ACCUM,
        # warmup_ratio=WARMUP_RATIO,
        warmup_steps=400,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        weight_decay=WD,
        bf16=BF16,
        # fp16=True,
        logging_steps=LOGGING_STEPS,
        output_dir=OUTPUT_DIR,
        optim='paged_adamw_8bit',#"adamw_torch",
        save_strategy="steps",
        save_steps=steps_per_epoch, # Save every epoch
        save_total_limit=SAVE_TOTAL_LIMIT,
        # THE FIX: `evaluation_strategy` was renamed to `eval_strategy`
        eval_strategy="steps",
        eval_steps=steps_per_epoch, # Evaluate every epoch
        ddp_find_unused_parameters=False,
        max_grad_norm=0.5,
        report_to=REPORT_TO,
    )

    data_collator = CustomDataCollator(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        # The trainer needs the Unsloth-patched tokenizer
        tokenizer=tokenizer,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        data_collator=data_collator,
        args=args,
    )
    
    if USE_GRAD_CHKPT:
        trainer.model.gradient_checkpointing_enable()

    # --- Train ---
    trainer.train()

    # --- Save Final Model ---
    print(f"Saving final LoRA adapters to {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)


if __name__ == "__main__":
    if REPORT_TO == "wandb":
        wandb.login()
    main()