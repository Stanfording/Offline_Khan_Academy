import os, json, time, math, random, re
import numpy as np
import torch
from typing import List, Dict, Any, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, get_scheduler
from peft import PeftModel
import wandb

# --- Config ---
BASE_MODEL = os.getenv("BASE_MODEL", "./unsloth_gemma3n_sft")  # SFT output as starting policy
REF_MODEL  = os.getenv("REF_MODEL",  "./unsloth_gemma3n_sft")  # start with same weights
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./gppo_gemma3n_rlhf")
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY", "")
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "mr_delight_rl")
SEED = 42
DEVICE = "cuda"
GROUP_SIZE = int(os.getenv("GROUP_SIZE","4"))
LR = float(os.getenv("LR","1e-6"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS","400"))
KL_COEFF = float(os.getenv("KL_COEFF","0.02"))
CLIP_RANGE = float(os.getenv("CLIP_RANGE","0.2"))
ITERS_PER_SAMPLE = int(os.getenv("ITERS_PER_SAMPLE","2"))
SAVE_EVERY = int(os.getenv("SAVE_EVERY","50"))

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

# --- Prompt tools ---
SYSTEM_PROMPT = open("distilled_prompt.txt","r",encoding="utf-8").read()

def build_user_from_sample(sample: Dict[str,Any]) -> str:
    user_action = sample["messages"][1]["content"]
    return user_action

def to_prompt(sample: Dict[str,Any]) -> str:
    system = SYSTEM_PROMPT
    user = build_user_from_sample(sample)
    return f"<|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>\n"

def ensure_json(s: str) -> bool:
    try:
        obj = json.loads(s)
        if isinstance(obj, dict) and "actions" in obj:
            return True
        return False
    except Exception:
        return False

SELF_Q = re.compile(r"\?|why|how|what if|so then|does that mean", re.I)
def has_self_questioning(text: str) -> bool:
    return bool(SELF_Q.search(text))

def reward_from_gemini(judger_input: Dict[str,Any]) -> float:
    """
    Simple external judge using Gemini via genai.
    We’ll prompt: reward = 1 if JSON-only and contains self-question, else 0.2; penalize broken schema.
    """
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        m = genai.GenerativeModel("gemini-2.0-flash-exp")
        prompt = (
            "Evaluate assistant output quality for Mr. Delight.\n"
            "Rules: Must be strict JSON starting with {\"actions\":[...]}, use allowed commands, "
            "and include at least one self-question in notes. Return a single float between 0 and 1.\n\n"
            f"User Input:\n{judger_input['user']}\n\nAssistant Output:\n{judger_input['assistant']}"
        )
        res = m.generate_content(prompt)
        txt = (res.text or "").strip()
        # Try parse float in text
        m = re.search(r"([01](?:\.\d+)?)", txt)
        if m:
            val = float(m.group(1))
            return max(0.0, min(1.0, val))
    except Exception:
        pass
    # Offline heuristic fallback
    asst = judger_input["assistant"]
    base = 0.2
    if ensure_json(asst):
        base += 0.4
    if has_self_questioning(asst):
        base += 0.3
    return float(max(0.0, min(1.0, base)))

def tokenize_texts(texts: List[str], tokenizer, device) -> Tuple[torch.Tensor, torch.Tensor]:
    toks = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    return toks["input_ids"].to(device), toks["attention_mask"].to(device)

def calc_logprob_sum(model, tokenizer, input_ids, attn_mask, requires_grad=False):
    labels = input_ids.clone()
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=pad_id, reduction="none")
    ctx = torch.enable_grad() if requires_grad else torch.no_grad()
    with ctx:
        out = model(input_ids=input_ids, attention_mask=attn_mask)
        logits = out.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous().to(shift_logits.device)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss_per_token = loss.view(shift_labels.shape)
        mask = shift_labels != pad_id
        nll = (loss_per_token * mask).sum(dim=1)
        lp = -nll
        return lp

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    wandb.init(project=WANDB_PROJECT, name=f"gppo_gemma3n_{int(time.time())}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    policy = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    ref = AutoModelForCausalLM.from_pretrained(
        REF_MODEL, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    ).eval()

    opt = torch.optim.AdamW(policy.parameters(), lr=LR)
    # Just a linear schedule over a short run
    sched = get_scheduler("linear", optimizer=opt, num_warmup_steps=0, num_training_steps=10000)

    # Load a small RL pool from train/dev/test chains (shuffled)
    from data_prep import load_chain_jsonl, to_chat_sample
    pool = []
    for p in [
        "mn_dataset_out_v6_trust/clean/train_chains.jsonl",
        "mn_dataset_out_v6_trust/clean/dev_chains.jsonl",
        "mn_dataset_out_v6_trust/clean/test_chains.jsonl",
    ]:
        pool.extend(load_chain_jsonl(p))
    random.shuffle(pool)
    # Convert to chat samples (system/user/assistant) and keep only user turns as prompts
    rl_samples = [to_chat_sample(x) for x in pool]
    print(f"RL pool size: {len(rl_samples)}")

    gen_cfg = GenerationConfig(
        temperature=0.9, top_p=0.95, do_sample=True,
        max_new_tokens=MAX_NEW_TOKENS,
        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
    )

    global_step = 0
    for i, sample in enumerate(rl_samples):
        user = sample["messages"][1]["content"]
        prompt = to_prompt(sample)

        # Generate GROUP_SIZE samples
        inputs = tokenizer([prompt]*GROUP_SIZE, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
        policy.eval()
        with torch.no_grad():
            outs = policy.generate(
                **inputs, temperature=gen_cfg.temperature, top_p=gen_cfg.top_p,
                do_sample=True, max_new_tokens=gen_cfg.max_new_tokens,
                pad_token_id=gen_cfg.pad_token_id, eos_token_id=gen_cfg.eos_token_id
            )
        dec = tokenizer.batch_decode(outs, skip_special_tokens=True)

        # Extract only assistant part after the prompt
        prompt_str = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
        completions = [d[len(prompt_str):] for d in dec]

        # Scores via Gemini judge (or heuristic fallback)
        rewards = []
        for c in completions:
            judger_input = {"user": user, "assistant": c.strip()}
            r = reward_from_gemini(judger_input)
            rewards.append(r)
        rewards = np.array(rewards, dtype=np.float32)

        # Compute old/policy logprobs on generated sequences (teacher forcing)
        # Re-tokenize the full sequences (prompt+completion) so logprobs are well-defined
        full_texts = [prompt + c for c in completions]
        input_ids, attn = tokenize_texts(full_texts, tokenizer, DEVICE)
        policy_logp = calc_logprob_sum(policy, tokenizer, input_ids, attn, requires_grad=True)
        ref_logp = calc_logprob_sum(ref, tokenizer, input_ids, attn, requires_grad=False)

        # Advantages
        adv = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        adv_t = torch.tensor(adv, dtype=torch.float32, device=policy_logp.device)

        ratio = torch.exp(policy_logp - ref_logp.detach())
        pg_unclipped = ratio * adv_t
        pg_clipped = torch.clamp(ratio, 1.0-CLIP_RANGE, 1.0+CLIP_RANGE) * adv_t
        pg_loss = -torch.min(pg_unclipped, pg_clipped).mean()

        # KL penalty (policy vs ref)
        kl_div = (policy_logp - ref_logp).mean()
        kl_loss = KL_COEFF * kl_div
        loss = pg_loss + kl_loss

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        sched.step()

        global_step += 1
        wandb.log({
            "step": global_step,
            "reward_mean": float(rewards.mean()),
            "reward_std": float(rewards.std()),
            "pg_loss": float(pg_loss.item()),
            "kl_div": float(kl_div.item()),
            "loss": float(loss.item()),
        })

        if global_step % SAVE_EVERY == 0:
            save_dir = os.path.join(OUTPUT_DIR, f"step_{global_step}")
            os.makedirs(save_dir, exist_ok=True)
            policy.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            print(f"[RL] saved {save_dir}")

    # Final save
    policy.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"[RL] finished, saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()