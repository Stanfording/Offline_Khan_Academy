import os, json, math, random
from typing import Dict, Any
from datasets import Dataset
from transformers import AutoTokenizer, TrainingArguments
from unsloth import FastLanguageModel
from peft import LoraConfig

from data_prep import build_chat_dataset

# Env/config
BASE_MODEL = os.getenv("BASE_MODEL", "google/gemma-2-9b-it")  # replace with gemma3n:e4b path if on HF or local
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./unsloth_gemma3n_sft")
TRAIN_PATH = os.getenv("TRAIN_PATH", "mn_dataset_out_v6_trust/clean/train_chains.jsonl")
DEV_PATH   = os.getenv("DEV_PATH",   "mn_dataset_out_v6_trust/clean/dev_chains.jsonl")
SEED = int(os.getenv("SEED", "42"))
MAX_SEQ_LEN = int(os.getenv("MAX_SEQ_LEN", "2048"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1"))
GRAD_ACCUM = int(os.getenv("GRAD_ACCUM", "16"))
LR = float(os.getenv("LR", "2e-5"))
EPOCHS = float(os.getenv("EPOCHS", "2.0"))
USE_QLORA = os.getenv("USE_QLORA", "true").lower() == "true"

random.seed(SEED)

def format_chat(example: Dict[str, Any], tokenizer):
    """
    Turn messages into a single prompt for causal LM.
    We’ll concatenate: system + user + assistant and train next-token prediction.
    """
    messages = example["messages"]
    # Simple format (you can use chat templates if the tokenizer supports it)
    sys = messages[0]["content"]
    usr = messages[1]["content"]
    asst = messages[2]["content"]
    prompt = f"<|system|>\n{sys}\n<|user|>\n{usr}\n<|assistant|>\n"
    target = asst
    full = prompt + target
    input_ids = tokenizer(full, truncation=True, max_length=MAX_SEQ_LEN)["input_ids"]
    labels = input_ids[:]  # train all tokens; optional: mask prompt tokens if desired
    return {"input_ids": input_ids, "labels": labels}

def main():
    # Build dataset
    train_chats, dev_chats = build_chat_dataset(TRAIN_PATH, DEV_PATH)
    train_ds = Dataset.from_list(train_chats)
    eval_ds = Dataset.from_list(dev_chats) if dev_chats else None

    # Load model with Unsloth
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = BASE_MODEL,
        max_seq_length = MAX_SEQ_LEN,
        load_in_4bit = USE_QLORA,
        dtype = None,
        device_map = "auto",
        trust_remote_code = True,
    )

    # LoRA config
    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj","v_proj","k_proj","o_proj","gate_proj","up_proj","down_proj"], # adjust to gemma3n blocks if needed
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = FastLanguageModel.get_peft_model(model, lora)

    # Tokenize
    def tok_map(ex):
        return format_chat(ex, tokenizer)
    train_tok = train_ds.map(tok_map, remove_columns=train_ds.column_names, num_proc=4)
    eval_tok = None
    if eval_ds:
        eval_tok = eval_ds.map(tok_map, remove_columns=eval_ds.column_names, num_proc=4)

    # Training args
    steps_per_epoch = math.ceil(len(train_tok) / (BATCH_SIZE * GRAD_ACCUM))
    train_steps = int(EPOCHS * steps_per_epoch)

    args = TrainingArguments(
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        num_train_epochs=EPOCHS,
        logging_steps=20,
        save_steps=max(100, steps_per_epoch),
        evaluation_strategy="steps" if eval_tok else "no",
        eval_steps=max(200, steps_per_epoch) if eval_tok else None,
        output_dir=OUTPUT_DIR,
        bf16=True,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        report_to="none",
    )

    FastLanguageModel.get_trainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        tokenizer=tokenizer,
    ).train()

    # Save PEFT
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"SFT artifacts saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()