import os, json, math, random, time, subprocess, sys
from typing import Dict, Any, Tuple, List
import numpy as np


os.environ["TE_DISABLE"] = "1"
os.environ["ACCELERATE_DISABLE_TE"] = "1"
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "transformer-engine", "transformer_engine"], check=False)
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
# Optional: in-container requirements installation (SageMaker)
# if os.environ.get("INSTALL_REQS","true").lower() == "true":
#     req = os.path.join(os.path.dirname(__file__), "requirements.txt")
#     if os.path.exists(req):
#         print("Installing requirements in-container...")
#         subprocess.run(["pip", "install", "-r", req], check=False)
#         print("Requirements installed.")

# def pip(args: list[str], check=True):
#     return subprocess.run([sys.executable, "-m", "pip"] + args, check=check)

# # One-time installs guarded by a flag to avoid re-install in retry loops
# if os.environ.get("BOOTSTRAPPED_UNSLOTH", "0") != "1":
#     try:
#         print("[BOOTSTRAP] Upgrading pip...")
        # pip(["install", "-U", "pip", "setuptools", "wheel"])

        # Choose one torch triplet. If your container is PyTorch 2.4, use these:
        # TORCH = "torch==2.3.0"
        # TV    = "torchvision==0.19.0"
        # TA    = "torchaudio==2.4.0"

        # print("[BOOTSTRAP] Installing core Torch triplet...")
        # pip(["install", "--no-cache-dir", TORCH, TV, TA])

        # Remove any preinstalled Transformer Engine to prevent import
        # print("[BOOTSTRAP] Removing transformer-engine if present...")
        # pip(["uninstall", "-y", "transformer-engine", "transformer_engine"], check=False)

        # # Install Unsloth (newest) + companions per Unsloth guidance
        # print("[BOOTSTRAP] Installing Unsloth (latest git) and required libs...")
        # pip([
        #     "install", "--no-cache-dir", "--no-deps",
        #     "git+https://github.com/unslothai/unsloth.git#egg=unsloth[colab-new]"
        # ])
        # pip([
        #     "install", "--no-cache-dir", "--no-deps",
        #     "git+https://github.com/unslothai/unsloth-zoo.git"
        # ])
        # # Required companion pins for newest Unsloth
        # pip(["install", "--no-cache-dir", "--force-reinstall", "--no-deps",
        #      "xformers<0.0.27", "trl<0.9.0", "peft", "accelerate", "bitsandbytes"])

        # # Your preferred versions (ensure compatible with Unsloth & Torch 2.4)
        # pip(["install", "--no-cache-dir",
        #      "transformers==4.43.3", "datasets==2.19.0",
        #      "peft==0.11.1", "accelerate==0.32.1", "bitsandbytes==0.43.1",
        #      "sentencepiece==0.2.0", "protobuf==4.25.8", "wandb==0.17.4",
        #      "boto3==1.34.129", "sagemaker==2.219.0", "python-dotenv==1.0.1"])

        # # Flash-Attention for Gemma 2 softcapping (no-deps to avoid torch reinstall)
        # print("[BOOTSTRAP] Installing flash-attn>=2.6.3 ...")
        # pip(["install", "--no-deps", "--no-cache-dir", "flash-attn>=2.6.3"])

        # os.environ["BOOTSTRAPPED_UNSLOTH"] = "1"
        # print("[BOOTSTRAP] Done.")
    # except subprocess.CalledProcessError as e:
    #     print(f"[BOOTSTRAP] Pip failed with return code {e.returncode}. Check logs.", flush=True)
    #     raise
from unsloth import FastLanguageModel
from datasets import Dataset
from transformers import TrainingArguments, DataCollatorForLanguageModeling
import torch
from peft import LoraConfig, PeftModel
import wandb

from data_prep import build_chat_dataset

# -----------------------
# Config
# -----------------------
BASE_MODEL = os.getenv("BASE_MODEL", "unsloth/gemma-3n-E4B")  # or "google/gemma3n:e4b"
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./unsloth_gemma3n_sft")
TRAIN_PATH = os.getenv("TRAIN_PATH", "mn_dataset_out_v6_trust/clean/train_chains.jsonl")
DEV_PATH   = os.getenv("DEV_PATH",   "mn_dataset_out_v6_trust/clean/dev_chains.jsonl")
SEED = int(os.getenv("SEED", "42"))
MAX_SEQ_LEN = int(os.getenv("MAX_SEQ_LEN", "4096"))
PER_DEVICE_BS = int(os.getenv("BATCH_SIZE", "1"))         # per-GPU
GRAD_ACCUM = int(os.getenv("GRAD_ACCUM", "16"))
LR = float(os.getenv("LR", "2e-5"))
EPOCHS = float(os.getenv("EPOCHS", "2.0"))
USE_QLORA = os.getenv("USE_QLORA", "true").lower() == "true"
USE_GRAD_CHKPT = os.getenv("USE_GRAD_CHKPT", "true").lower() == "true"
BF16 = os.getenv("BF16", "true").lower() == "true"
WD = float(os.getenv("WEIGHT_DECAY", "0.0"))
WARMUP_RATIO = float(os.getenv("WARMUP_RATIO", "0.03"))
LOGGING_STEPS = int(os.getenv("LOGGING_STEPS", "20"))
SAVE_TOTAL_LIMIT = int(os.getenv("SAVE_TOTAL_LIMIT", "3"))
REPORT_TO = os.getenv("REPORT_TO", "wandb")  # "wandb" | "tensorboard" | "none"
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "gemma3n_mr_delight_sft")
WANDB_RUN = os.getenv("WANDB_RUN", f"sft_{int(time.time())}")
MASK_PROMPT_LOSS = os.getenv("MASK_PROMPT_LOSS", "true").lower() == "true"
EVAL_JSON_VALIDITY_N = int(os.getenv("EVAL_JSON_VALIDITY_N", "128"))
MERGE_LORA_AT_END = os.getenv("MERGE_LORA_AT_END", "false").lower() == "true"

# -----------------------
# Reproducibility
# -----------------------
def seed_all(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
seed_all(SEED)

# -----------------------
# Formatting and masking
# -----------------------
def build_texts(example: Dict[str, Any]) -> Tuple[str, str]:
    messages = example["messages"]
    sys = messages[0]["content"]
    usr = messages[1]["content"]
    asst = messages[2]["content"]
    prompt = f"<|system|>\n{sys}\n<|user|>\n{usr}\n<|assistant|>\n"
    target = asst
    return prompt, target

def tokenize_mask(example: Dict[str, Any], tokenizer, max_len: int, mask_prompt: bool) -> Dict[str, Any]:
    prompt, target = build_texts(example)
    full = prompt + target
    toks = tokenizer(full, truncation=True, max_length=max_len)
    input_ids = toks["input_ids"]
    labels = input_ids.copy()
    if mask_prompt:
        prompt_ids = tokenizer(prompt, truncation=True, max_length=max_len)["input_ids"]
        prompt_len = len(prompt_ids)
        labels[:prompt_len] = [-100] * min(prompt_len, len(labels))
    return {"input_ids": input_ids, "labels": labels}

def is_valid_json(s: str) -> bool:
    try:
        obj = json.loads(s)
        return isinstance(obj, dict) and "actions" in obj
    except Exception:
        return False

def render_and_score_json_validity(policy, tokenizer, dev_raw: List[Dict[str, Any]], n_samples: int = 128) -> float:
    if not dev_raw:
        return float("nan")
    subset = min(n_samples, len(dev_raw))
    policy.eval()
    ok = 0
    with torch.no_grad():
        for ex in dev_raw[:subset]:
            prompt, _ = build_texts(ex)
            inputs = tokenizer(prompt, return_tensors="pt").to(policy.device)
            gen = policy.generate(
                **inputs,
                do_sample=True, temperature=0.7, top_p=0.95,
                max_new_tokens=512,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            out = tokenizer.decode(gen[0], skip_special_tokens=True)
            completion = out[len(prompt):] if out.startswith(prompt) else out
            ok += 1 if is_valid_json(completion.strip()) else 0
    return ok / subset

# -----------------------
# Main
# -----------------------
def main():
    # Data
    train_chats, dev_chats = build_chat_dataset(TRAIN_PATH, DEV_PATH)
    train_ds = Dataset.from_list(train_chats)
    eval_ds = Dataset.from_list(dev_chats) if dev_chats else None

    # Model
    # IMPORTANT for multi-GPU DDP: device_map=None so each rank keeps the model on its own GPU.
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=USE_QLORA,
        dtype=None,
        device_map=None,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if USE_GRAD_CHKPT and hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable()
            print("Gradient checkpointing enabled.")
        except Exception as e:
            print(f"Gradient checkpointing not enabled: {e}")

    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = FastLanguageModel.get_peft_model(model, lora)

    # Tokenize
    def tok_map(ex): return tokenize_mask(ex, tokenizer, MAX_SEQ_LEN, MASK_PROMPT_LOSS)
    num_proc = max(1, min(8, os.cpu_count() or 1))
    train_tok = train_ds.map(tok_map, remove_columns=train_ds.column_names, num_proc=num_proc)
    eval_tok = eval_ds.map(tok_map, remove_columns=eval_ds.column_names, num_proc=num_proc) if eval_ds else None

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    steps_per_epoch = max(1, math.ceil(len(train_tok) / (PER_DEVICE_BS * GRAD_ACCUM)))
    save_steps = max(100, steps_per_epoch)
    eval_steps = max(200, steps_per_epoch) if eval_tok else None

    # W&B
    if REPORT_TO == "wandb":
        wandb.init(project=WANDB_PROJECT, name=WANDB_RUN, config={
            "base_model": BASE_MODEL,
            "epochs": EPOCHS, "lr": LR,
            "per_device_bs": PER_DEVICE_BS, "grad_accum": GRAD_ACCUM,
            "qlora": USE_QLORA, "max_seq_len": MAX_SEQ_LEN,
            "mask_prompt_loss": MASK_PROMPT_LOSS
        })

    args = TrainingArguments(
        per_device_train_batch_size=PER_DEVICE_BS,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        num_train_epochs=EPOCHS,
        weight_decay=WD,
        logging_steps=LOGGING_STEPS,
        save_steps=save_steps,
        save_total_limit=SAVE_TOTAL_LIMIT,
        evaluation_strategy="steps" if eval_tok else "no",
        eval_steps=eval_steps,
        output_dir=OUTPUT_DIR,
        bf16=BF16,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        warmup_ratio=WARMUP_RATIO,
        report_to=REPORT_TO if REPORT_TO in ("wandb","tensorboard") else "none",
        dataloader_num_workers=2,
        gradient_checkpointing=USE_GRAD_CHKPT,
        ddp_find_unused_parameters=False,   # critical for LoRA + DDP
        torch_compile=False,
    )

    trainer = FastLanguageModel.get_trainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    trainer.train()

    # Optional dev metric (JSON validity)
    if eval_ds:
        score = render_and_score_json_validity(model, tokenizer, dev_chats, n_samples=EVAL_JSON_VALIDITY_N)
        print(f"[Dev] JSON validity: {score:.3f}")
        if REPORT_TO == "wandb":
            wandb.log({"dev_json_validity_final": score})

    # Save adapters
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"SFT artifacts saved to {OUTPUT_DIR}")

    # Optional: merge LoRA for deployment
    if MERGE_LORA_AT_END:
        print("Merging LoRA into base model...")
        base, _ = FastLanguageModel.from_pretrained(
            model_name=BASE_MODEL, max_seq_length=MAX_SEQ_LEN,
            load_in_4bit=False, dtype="bfloat16" if BF16 else "float16",
            device_map="auto", trust_remote_code=True
        )
        peft_model = PeftModel.from_pretrained(base, OUTPUT_DIR)
        merged = peft_model.merge_and_unload()
        merge_out = os.path.join(OUTPUT_DIR, "merged")
        os.makedirs(merge_out, exist_ok=True)
        merged.save_pretrained(merge_out)
        tokenizer.save_pretrained(merge_out)
        print(f"Merged weights saved to {merge_out}")

    if REPORT_TO == "wandb":
        wandb.finish()

if __name__ == "__main__":
    main()