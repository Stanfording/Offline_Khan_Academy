# train.py

import os, json, math, random, time, sys
from typing import Dict, Any, Tuple, List
from functools import partial

# ============================ START: UNSLOTH BUG FIX v5 (FINAL) ============================
# This is the definitive fix for a race condition in Unsloth's TRL patching that
# crashes multi-GPU training on some environments (like SageMaker).
#
# STRATEGY: We create a fake "imposter" module for `unsloth.models.rl` and
# insert it into Python's module cache (`sys.modules`) *before* the main
# `import unsloth` happens.
#
# This imposter module satisfies the import dependencies from other parts of
# Unsloth but contains harmless, empty functions. The original, buggy `rl.py`
# file is never actually executed.

from types import ModuleType

# Create a new, empty module object
fake_rl_module = ModuleType("unsloth.models.rl")

# The other Unsloth files expect this function to exist. We provide a dummy one.
def dummy_PatchFastRL(*args, **kwargs):
    pass # Do nothing

# The buggy code is inside this function. We also provide a dummy.
def dummy_patch_trl_rl_trainers(*args, **kwargs):
    pass # Do nothing

# Add the dummy functions to our fake module
fake_rl_module.PatchFastRL = dummy_PatchFastRL
fake_rl_module.patch_trl_rl_trainers = dummy_patch_trl_rl_trainers

class dummy_vLLMSamplingParams:
    pass

fake_rl_module.vLLMSamplingParams = dummy_vLLMSamplingParams

# Insert the fake module into the system's module cache.
# This must happen BEFORE `import unsloth` is called.
sys.modules["unsloth.models.rl"] = fake_rl_module
print("✅ Unsloth multi-GPU bug fix applied: Imposter module injected.")
# ============================= END: UNSLOTH BUG FIX v5 (FINAL) =============================


# The rest of your script follows, unchanged.
import os, json, math, random, time
from typing import Dict, Any, Tuple, List
from functools import partial

# --- Imports ---
from unsloth import FastLanguageModel
# ... and so on

# --- Environment Toggles for Stability ---
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ.setdefault("TE_DISABLE", "1")
os.environ.setdefault("ACCELERATE_DISABLE_TE", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

# --- Imports ---
from unsloth import FastLanguageModel

from datasets import Dataset
from transformers import TrainingArguments, DataCollatorForLanguageModeling, AutoTokenizer, Trainer, default_data_collator, TrainerCallback, TrainerState, TrainerControl
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
GRAD_ACCUM = int(os.getenv("GRAD_ACCUM", "8"))
LR = float(os.getenv("LR", "2e-5"))
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
LOGGING_STEPS = int(os.getenv("LOGGING_STEPS", "2"))
SAVE_TOTAL_LIMIT = int(os.getenv("SAVE_TOTAL_LIMIT", "3"))
REPORT_TO = os.getenv("REPORT_TO", "wandb")

num_proc = 8

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
    """
    Tokenizes prompt+target, and creates labels with loss masking on the prompt.
    Includes a safety check to skip examples where the prompt is too long.
    """
    prompt, target = build_texts(example)
    if not prompt or not target:
        return {"input_ids": [], "labels": []}

    # Tokenize the prompt-only part to find its length
    toks_prompt = tokenizer(prompt, add_special_tokens=False)
    prompt_len = len(toks_prompt["input_ids"])
    
    # Tokenize the target
    toks_target = tokenizer(target + tokenizer.eos_token, add_special_tokens=False)
    target_len = len(toks_target["input_ids"])

    # === SAFETY CHECK ===
    # If the prompt alone is already filling the context, the target was cut off. Skip this example.
    if prompt_len >= max_len:
        # Optionally, print a warning to see how often this happens
        # print(f"WARNING: Skipping example. Prompt length ({prompt_len}) exceeds max_len ({max_len}).")
        return {"input_ids": [], "labels": [], "attention_mask": []}
    
    # Combine and truncate
    input_ids = toks_prompt["input_ids"] + toks_target["input_ids"]
    if len(input_ids) > max_len:
        input_ids = input_ids[:max_len]
        # Adjust target_len if it was truncated
        target_len = max_len - prompt_len

    labels = list(input_ids)
    
    if mask_prompt:
        # Mask the prompt part
        for i in range(prompt_len):
            labels[i] = -100

    # Make sure we didn't accidentally mask everything
    # This can happen if the combined text was truncated exactly at the prompt's end
    if all(l == -100 for l in labels):
        return {"input_ids": [], "labels": [], "attention_mask": []}

    attention_mask = [1] * len(input_ids)

    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}

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

class PredictionCallback(TrainerCallback):
    """
    A callback that logs a model's prediction on a fixed validation example.
    This is useful for visually inspecting the model's generation quality over time.
    """
    def __init__(self, eval_dataset, tokenizer):
        super().__init__()
        # Store a single example from the validation set
        self.eval_example = eval_dataset[0]
        self.tokenizer = tokenizer

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Hook called at the end of every evaluation phase.
        """
        # The model is passed in kwargs during evaluation
        model = kwargs['model']
        
        print(f"\n\n--- Generating Sample at Step {state.global_step} ---")

        # 1. Prepare the Input (Prompt)
        # Find where the prompt ends and the label begins (the first non -100 label)
        try:
            prompt_end_index = self.eval_example['labels'].index(next(l for l in self.eval_example['labels'] if l != -100))
        except StopIteration:
            prompt_end_index = len(self.eval_example['input_ids'])

        prompt_input_ids = self.eval_example['input_ids'][:prompt_end_index]
        
        # Move input_ids to the same device as the model
        prompt_tensor = torch.tensor(prompt_input_ids).unsqueeze(0).to(model.device)
        attention_mask = torch.ones_like(prompt_tensor) # Create an attention mask of all 1s

        # 2. Generate the Model's Output
        generated_ids = model.generate(
            prompt_tensor,
            max_new_tokens=1024, # Limit generation length
            do_sample=True,
            temperature=1,
            top_p=1,
            attention_mask=attention_mask,
        )
        
        # 3. Decode for Visual Comparison
        # Decode the prompt
        prompt_text = self.tokenizer.decode(prompt_tensor[0], skip_special_tokens=True)
        
        # Decode the ground truth target
        # Filter out the -100 labels before decoding
        ground_truth_ids = [label for label in self.eval_example['labels'] if label != -100]
        ground_truth_text = self.tokenizer.decode(ground_truth_ids, skip_special_tokens=True)

        # Decode the model's generated output (only the new tokens)
        model_output_text = self.tokenizer.decode(generated_ids[0][len(prompt_tensor[0]):], skip_special_tokens=True)

        # 4. Print the comparison
        print("\n--- PROMPT ---")
        print(prompt_text)
        print("\n--- GROUND TRUTH (TARGET) ---")
        print(ground_truth_text)
        print("\n--- MODEL OUTPUT ---")
        print(model_output_text)
        print("\n--------------------------------------------------\n")

        # Optional: Log to WandB as a table for easy comparison across steps
        if "wandb" in args.report_to:
            table = wandb.Table(columns=["step", "prompt", "ground_truth", "model_output"])
            table.add_data(state.global_step, prompt_text, ground_truth_text, model_output_text)
            wandb.log({"predictions": table})

# ==============================================================================
# 3. Main Training Orchestration
# ==============================================================================
def main():

    def get_local_rank():
        lr = os.environ.get("LOCAL_RANK")
        return int(lr) if lr is not None else 0

    local_rank = get_local_rank()
    if local_rank == 0 and REPORT_TO == "wandb":
        wandb.init(project="Mr-Delight")

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

    
    train_ds = Dataset.from_list(train_chats).filter(example_is_valid, num_proc=num_proc)
    eval_ds = Dataset.from_list(dev_chats).filter(example_is_valid, num_proc=num_proc)
    
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
    
    train_tok = train_ds.map(tok_map_fn, remove_columns=train_ds.column_names, num_proc=num_proc)
    eval_tok = eval_ds.map(tok_map_fn, remove_columns=eval_ds.column_names, num_proc=num_proc)

    # ======================= DEBUGGING SNIPPET =======================
    print("🕵️  Inspecting a tokenized example from the training set...")
    try:
        example = train_tok[0]
        map_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
        if map_tokenizer.pad_token is None:
            map_tokenizer.pad_token = map_tokenizer.eos_token
            
        print(f"Input IDs length: {len(example['input_ids'])}")
        print(f"Labels length:    {len(example['labels'])}")
        
        non_masked_labels = [l for l in example['labels'] if l != -100]
        print(f"Number of non-masked labels (should be > 0): {len(non_masked_labels)}")
    
        if not non_masked_labels:
            print("🚨 CRITICAL: All labels in this example are masked! This is the cause of the 0.0 loss.")
            
            # Let's see why by decoding
            prompt_portion = [tok if lab == -100 else -1 for tok, lab in zip(example['input_ids'], example['labels'])]
            target_portion = [tok if lab != -100 else -1 for tok, lab in zip(example['input_ids'], example['labels'])]
    
            decoded_prompt = map_tokenizer.decode([t for t in prompt_portion if t != -1], skip_special_tokens=True)
            decoded_target = map_tokenizer.decode([t for t in target_portion if t != -1], skip_special_tokens=True)
    
            print("\n--- Decoded Prompt Portion (Masked) ---")
            print(decoded_prompt)
            print("\n--- Decoded Target Portion (Unmasked) ---")
            print(decoded_target)
            
    except Exception as e:
        print(f"Error during inspection: {e}")
    # =================================================================

    # --- Trainer Setup ---
    steps_per_epoch = 2 #math.ceil(len(train_tok) / (PER_DEVICE_BS * GRAD_ACCUM))
    
    args = TrainingArguments(
        per_device_train_batch_size=PER_DEVICE_BS,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=GRAD_ACCUM,
        # warmup_ratio=WARMUP_RATIO,
        warmup_steps=400,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        weight_decay=WD,
        bf16=BF16,
        fp16=False,
        logging_steps=LOGGING_STEPS,
        output_dir=OUTPUT_DIR,
        optim= 'adamw_torch', #'paged_adamw_8bit',#"adamw_torch",
        save_strategy="steps",
        save_steps=steps_per_epoch, # Save every epoch
        save_total_limit=SAVE_TOTAL_LIMIT,
        # THE FIX: `evaluation_strategy` was renamed to `eval_strategy`
        eval_strategy="steps",
        eval_steps=steps_per_epoch, # Evaluate every epoch
        ddp_find_unused_parameters=False,
        max_grad_norm=0.5,
        report_to=REPORT_TO,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        # optim_args="eps=1e-7",
    )

    data_collator = CustomDataCollator(tokenizer=tokenizer, mlm=False)
    prediction_callback = PredictionCallback(eval_dataset=eval_tok, tokenizer=map_tokenizer)


    trainer = Trainer(
        model=model,
        # The trainer needs the Unsloth-patched tokenizer
        tokenizer=map_tokenizer,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        data_collator=data_collator,
        args=args,
        callbacks=[prediction_callback],
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