import os
import json
import uuid
import random
import hashlib
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from dotenv import load_dotenv
load_dotenv()

import google.generativeai as genai
from google.generativeai import types as gm_types

# -----------------------------
# CONFIG
# -----------------------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError("Set GOOGLE_API_KEY in your environment or .env")

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
PROMPT_PATH = os.getenv("DELIGHT_PROMPT_PATH", "./prompt3_2.txt")
OUT_DIR = os.getenv("OUT_DIR", "./mn_dataset_out_v3")
SEED = int(os.getenv("SEED", "42"))
random.seed(SEED)

# Total counts (adjust with your budget)
TARGET_COUNTS = {
    "train": 2000 * 3,
    "dev": 240,
    "test": 240,
}

# Chain ratio (subset of data are short multi-turn chains)
CHAIN_RATIO = float(os.getenv("CHAIN_RATIO", "0.9"))  # 25% chains (3–5 steps)

# Rate/concurrency
MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT", "8"))
SLEEP_BETWEEN_BATCH = float(os.getenv("SLEEP_BETWEEN_BATCH", "0.45"))

# --------------------------------
# SUBJECTS/TOPICS
# --------------------------------
GRADES = [
    "K", "1st", "2nd", "3rd", "4th", "5th",
    "6th", "7th", "8th", "9th", "10th", "11th", "12th",
    "Intro College"
]

DOMAINS: Dict[str, List[str]] = {
  "Math": [
    "Counting and Place Value", "Addition/Subtraction Basics", "Multiplication/Division Basics",
    "Fractions & Decimals", "Ratios & Proportions", "Percentages & Interest",
    "Geometry Basics", "Area/Perimeter", "Volume & Surface Area",
    "Statistics & Data", "Probability Basics",
    "Linear Equations", "Systems of Equations", "Inequalities",
    "Slope & Graphing Lines", "Quadratics & Factoring",
    "Exponents & Radicals", "Polynomials",
    "Trigonometry (Intro)", "Limits (Intro)", "Derivatives (Intro)", "Integrals (Intro)"
  ],
  "Physics": [
    "Units & Measurement", "Motion (1D/2D)", "Forces & Newton's Laws",
    "Energy & Work", "Momentum & Collisions",
    "Circular Motion & Gravitation", "Waves & Sound", "Light & Optics",
    "Electricity (Basics)", "Circuits (DC)", "Magnetism (Intro)"
  ],
  "Chemistry": [
    "Matter & States", "Atomic Structure", "Periodic Table",
    "Chemical Bonds", "Chemical Reactions", "Stoichiometry",
    "Acids & Bases", "Gases", "Thermochemistry",
    "Solutions & Concentration", "Equilibrium (Intro)"
  ],
  "Biology": [
    "Cell Structure", "Photosynthesis", "Cellular Respiration",
    "DNA/RNA & Protein Synthesis", "Genetics (Basics)",
    "Evolution & Natural Selection", "Human Body Systems",
    "Ecosystems & Food Webs", "Biodiversity & Conservation"
  ],
  "Computer Science": [
    "Algorithms (Sorting & Searching)", "Data Structures (Arrays/Lists/Stacks/Queues)",
    "Big-O (Intro)", "Recursion (Intro)", "Binary & Hex",
    "Boolean Logic", "Basic Python", "Basic JavaScript", "Control Flow",
    "Functions & Parameters", "Debugging Strategies"
  ],
  "Language Arts": [
    "Phonics & Sight Words", "Reading Comprehension",
    "Main Idea & Details", "Summarizing", "Inference",
    "Narrative Writing", "Persuasive Writing", "Grammar & Parts of Speech",
    "Punctuation", "Essay Structure", "Poetry (Intro)"
  ],
  "History": [
    "Ancient Civilizations", "Middle Ages", "Renaissance",
    "American Revolution", "US Constitution (Basics)",
    "Industrial Revolution", "World Wars Overview",
    "Civil Rights Movement", "Modern Globalization (Intro)"
  ],
  "Finance": [
    "Counting Money", "Budgeting Basics", "Savings vs. Spending",
    "Simple Interest", "Compound Interest (Intro)",
    "Taxes (Basics)", "Credit & Debt (Intro)"
  ],
  "Art & Music": [
    "Color Wheel & Mixing", "Perspective (Intro)", "Drawing Basics",
    "Rhythm & Beat", "Scales (Intro)", "Reading Music (Intro)"
  ],
  "Study Skills": [
    "Note-Taking Strategies", "Memory Techniques (Spaced Repetition)",
    "Test-Taking Strategies", "Goal Setting & Planning"
  ],
  "Maker/Practical": [
    "Cooking Measurements", "Unit Conversions", "Maps & Coordinates",
    "Intro Electronics (Circuits)", "3D Printing Basics",
    "Environmental Science Projects (Water Quality)"
  ],
  "Test Prep": [
    "SAT Math (Linear/Quadratic)", "SAT Reading (Passage Strategy)",
    "SAT Writing (Grammar)", "ACT Science (Data & Experiments)"
  ]
}

DOMAIN_TOPICS: List[Tuple[str,str]] = []
for d, topics in DOMAINS.items():
    for t in topics:
        DOMAIN_TOPICS.append((d, t))

# --------------------------------
# IO
# --------------------------------
out_dir = Path(OUT_DIR)
(out_dir / "raw").mkdir(parents=True, exist_ok=True)
(out_dir / "clean").mkdir(exist_ok=True)
(out_dir / "rejected").mkdir(exist_ok=True)

# --------------------------------
# Gemini init
# --------------------------------
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel(MODEL_NAME)

def read_prompt() -> str:
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read().strip()

SYSTEM_PROMPT = read_prompt()

# --------------------------------
# Helpers
# --------------------------------
def short_hash(obj: Any) -> str:
    s = json.dumps(obj, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:10]

def ensure_json(s: str) -> Optional[dict]:
    try:
        return json.loads(s)
    except Exception:
        start = s.find("{")
        if start < 0: return None
        stack = 0
        for i in range(start, len(s)):
            if s[i] == "{": stack += 1
            elif s[i] == "}":
                stack -= 1
                if stack == 0:
                    chunk = s[start:i+1]
                    try: return json.loads(chunk)
                    except Exception: return None
        return None

ALLOWED_COMMANDS = {
    "ui_display_notes","ui_short_answer","ui_mcq","ui_checkbox","ui_slider",
    "ui_button","ui_rearrange_order","ui_drawing_board","update_status"
}

def is_equation_like(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in ["solve", "equation", "factor", "simplify", "x =", "y =", "derivative", "integral", "slope", "graph"])

def validate_actions_payload(payload: dict) -> Tuple[bool, dict, str]:
    if not isinstance(payload, dict) or "actions" not in payload or not isinstance(payload["actions"], list):
        return False, payload, "Missing actions"

    repaired = []
    for a in payload["actions"]:
        if not isinstance(a, dict) or "command" not in a:
            return False, payload, "Bad action item"
        cmd = a["command"]
        if cmd not in ALLOWED_COMMANDS:
            return False, payload, f"Disallowed command: {cmd}"
        params = a.get("parameters", {})
        if not isinstance(params, dict):
            params = {}

        if cmd == "ui_short_answer":
            qtext = params.get("question_text","")
            var = params.get("variable_name","")
            if is_equation_like(qtext) or var.startswith("q_"):
                hints = params.get("hints", {})
                if not isinstance(hints, dict): hints = {}
                hints["math_input"] = True
                params["hints"] = hints

        if cmd == "update_status":
            ups = params.get("updates", {})
            if isinstance(ups, dict):
                stg = ups.get("lesson_stage")
                if stg == "intuition":
                    ups["current_lesson_progress"] = "10"
                elif stg == "main_lesson":
                    ups["current_lesson_progress"] = "40"
                elif stg == "practice":
                    ups["current_lesson_progress"] = "70"
                elif stg == "lesson_complete":
                    ups["current_lesson_progress"] = "100"
                params["updates"] = ups

        repaired.append({"command": cmd, "parameters": params})

    if not repaired:
        return False, payload, "Empty actions"
    return True, {"actions": repaired}, ""

def apply_updates_to_status(status: dict, action: dict) -> dict:
    """Merge update_status action into a new status dict (clamped)."""
    ups = action.get("parameters",{}).get("updates",{})
    new_status = json.loads(json.dumps(status))  # deep copy
    for k, v in ups.items():
        if "." in k:
            top, sub = k.split(".", 1)
            new_status.setdefault(top, {})
            if isinstance(v, str) and (v.startswith("+") or v.startswith("-")):
                try:
                    delta = float(v)
                    new_status[top][sub] = float(new_status[top].get(sub, 0.0)) + delta
                except: pass
            else:
                new_status[top][sub] = v
        else:
            if isinstance(v, str) and (v.startswith("+") or v.startswith("-")):
                try:
                    delta = float(v)
                    new_status[k] = float(new_status.get(k, 0.0)) + delta
                except: pass
            else:
                new_status[k] = v
    # Clamp
    for f in ["learning_confidence","learning_interest","learning_patience","effort_focus"]:
        if f in new_status and isinstance(new_status[f], (int,float)):
            new_status[f] = max(1, min(10, new_status[f]))
    if "current_lesson_progress" in new_status and isinstance(new_status["current_lesson_progress"], (int,float)):
        new_status["current_lesson_progress"] = max(0, min(100, new_status["current_lesson_progress"]))
    return new_status

def extract_first_update_status(actions: List[dict]) -> Optional[dict]:
    for a in actions:
        if a.get("command") == "update_status":
            return a
    return None

# --------------------------------
# Seed builders
# --------------------------------
SEED_KINDS = [
    "init","plan","start_intuition","paraphrase_good","paraphrase_weak",
    "main_checkpoint_easy","main_checkpoint_mid","main_checkpoint_hard",
    "summary_good","summary_weak",
    "practice_correct_easy","practice_incorrect_easy",
    "practice_correct_mid","practice_incorrect_mid",
    "practice_correct_hard","practice_incorrect_hard",
    "lesson_complete","handle_user_question","give_feedback","show_status"
]

def random_status(domain: str, topic: str) -> Dict[str, Any]:
    base = {
        "learning_confidence": random.randint(3,8),
        "learning_interest": random.randint(4,9),
        "learning_patience": random.randint(4,9),
        "effort_focus": random.randint(5,9),
        "weak_concept_spot": {},
        "current_lesson_progress": random.choice([0,10,40,70,90]),
        "current_lesson_title": f"{topic}",
        "lesson_stage": random.choice(["not_in_lesson","intuition","main_lesson","practice"])
    }
    # domain-aware weak spots
    if domain == "Math":
        base["weak_concept_spot"] = {random.choice(["Fractions","BalanceMoves","QuadraticForm","PrimeNumbers","Slopes"]): random.randint(3,7)}
    elif domain == "Physics":
        base["weak_concept_spot"] = {random.choice(["Forces","Kinematics","Circuits"]): random.randint(3,7)}
    elif domain == "Chemistry":
        base["weak_concept_spot"] = {random.choice(["Stoichiometry","AcidsBases","Bonds"]): random.randint(3,7)}
    elif domain == "Biology":
        base["weak_concept_spot"] = {random.choice(["Photosynthesis","Genetics","Ecosystems"]): random.randint(3,7)}
    return base

def build_single_seed() -> Dict[str, Any]:
    domain, topic = random.choice(DOMAIN_TOPICS)
    grade = random.choice(GRADES)
    seed_kind = random.choice(SEED_KINDS)
    status = random_status(domain, topic)

    def qid(prefix: str, diff: str = ""):
        d = f"{prefix}_{diff}" if diff else prefix
        return f"q_{domain[:2].lower()}_{topic[:6].lower()}_{d}_{random.randint(1,999)}"

    if seed_kind == "init":
        ua = {"command":"init","parameters":{"user_name":"Learner"}}
        # canonical init state
        status["lesson_stage"] = "not_in_lesson"
        status["current_lesson_progress"] = 0
    elif seed_kind == "plan":
        ua = {"command":"generate_course_plan","parameters":{"query":f"{domain}: {topic} for {grade}"}}
    elif seed_kind == "start_intuition":
        ua = {"command":"start_lesson","parameters":{"lesson_index":0}}
        status["lesson_stage"] = "not_in_lesson"
        status["current_lesson_progress"] = 0
    elif seed_kind == "paraphrase_good":
        ua = {"command":"evaluate_paraphrase","parameters":{"user_input": random.choice([
            "Main idea: balance the equation by doing the same to both sides.",
            "We use the story/analogy to frame the concept."
        ])}}
    elif seed_kind == "paraphrase_weak":
        ua = {"command":"evaluate_paraphrase","parameters":{"user_input": random.choice([
            "I’m not sure about the key idea.",
            "I think it’s about moving stuff but I’m confused."
        ])}}
    elif seed_kind in ["main_checkpoint_easy","main_checkpoint_mid","main_checkpoint_hard"]:
        diff = seed_kind.split("_")[-1]
        ua = {"command":"evaluate_answer","parameters":{
            "question_id": qid("checkpoint", diff),
            "user_answer": random.choice(["A","B","C","x=3","7"])
        }}
    elif seed_kind == "summary_good":
        ua = {"command":"evaluate_summary","parameters":{"user_input": "Clear steps, rule restated, example applied."}}
    elif seed_kind == "summary_weak":
        ua = {"command":"evaluate_summary","parameters":{"user_input": "I lost track of the steps."}}
    elif seed_kind.startswith("practice_correct"):
        diff = seed_kind.split("_")[-1]
        ua = {"command":"evaluate_answer","parameters":{"question_id": qid("lesson", diff), "user_answer": random.choice(["Correct","x=3","4","Area=12"])}}
    elif seed_kind.startswith("practice_incorrect"):
        diff = seed_kind.split("_")[-1]
        ua = {"command":"evaluate_answer","parameters":{"question_id": qid("lesson", diff), "user_answer": random.choice(["Incorrect","x=10","9","Area=5"])}}
    elif seed_kind == "lesson_complete":
        ua = {"command":"skip_practice","parameters":{}}
        status["lesson_stage"] = "practice"
        status["current_lesson_progress"] = 90
    elif seed_kind == "handle_user_question":
        ua = {"command":"handle_user_question","parameters":{"user_question_value": random.choice([
            "How does this connect to real life?",
            "Why do we divide here instead of subtract?"
        ])}}
    elif seed_kind == "give_feedback":
        ua = {"command":"give_feedback","parameters":{}}
    elif seed_kind == "show_status":
        ua = {"command":"show_status","parameters":{}}
    else:
        ua = {"command":"init","parameters":{"user_name":"Learner"}}
        status["lesson_stage"] = "not_in_lesson"
        status["current_lesson_progress"] = 0

    return {
        "user_action": ua,
        "status_dictionary": status,
        "seed_kind": seed_kind,
        "domain": domain,
        "grade": grade,
        "topic": topic
    }

def build_messages(seed_turn: dict) -> List[Dict[str, Any]]:
    user_json = json.dumps({
        "user_action": seed_turn["user_action"],
        "status_dictionary": seed_turn["status_dictionary"]
    })
    return [
        {"role":"user","parts":[
            SYSTEM_PROMPT,
            "Output strict JSON only: {\"actions\":[...]}",
            "Allowed commands only. Include update_status with lesson_stage/progress on stage changes.",
            "Use hints.math_input only for equation-like short answers.",
            "Route inputs: initial_paraphrase→evaluate_paraphrase, final_summary→evaluate_summary, q_*→evaluate_answer, user_feedback_input→process_feedback.",
            f"Next user turn:\n{user_json}"
        ]}
    ]

# --------------------------------
# Gemini call + validation
# --------------------------------
async def call_model_and_clean(turn_id: str, seed_turn: dict) -> Tuple[Optional[dict], dict, str]:
    msgs = build_messages(seed_turn)
    try:
        resp = model.generate_content(
            msgs,
            safety_settings={
                'HARM_CATEGORY_HARASSMENT':'BLOCK_NONE',
                'HARM_CATEGORY_HATE_SPEECH':'BLOCK_NONE',
                'HARM_CATEGORY_SEXUALLY_EXPLICIT':'BLOCK_NONE',
                'HARM_CATEGORY_DANGEROUS_CONTENT':'BLOCK_NONE'
            },
            generation_config=gm_types.GenerationConfig(
                temperature=0.8,
                candidate_count=1
            )
        )
        raw_text = resp.text or ""
    except Exception as e:
        return None, {"error": str(e)}, f"API error: {e}"

    raw_payload = ensure_json(raw_text)
    if not raw_payload:
        return None, {"raw": raw_text}, "Non-JSON"

    ok, repaired, reason = validate_actions_payload(raw_payload)
    if not ok:
        return None, raw_payload, f"Schema invalid: {reason}"

    cleaned = {
        "id": turn_id,
        "input": {
            "user_action": seed_turn["user_action"],
            "status_dictionary": seed_turn["status_dictionary"],
            "summary_context": f"{seed_turn['domain']} / {seed_turn['topic']} for {seed_turn['grade']} | Seed: {seed_turn['seed_kind']}"
        },
        "output": repaired,
        "meta": {
            "seed_kind": seed_turn["seed_kind"],
            "domain": seed_turn["domain"],
            "grade": seed_turn["grade"],
            "topic": seed_turn["topic"]
        }
    }
    return cleaned, raw_payload, ""

# --------------------------------
# Producers
# --------------------------------
async def producer_single(split: str, target_n: int):
    clean_path = Path(OUT_DIR) / "clean" / f"{split}.jsonl"
    raw_path = Path(OUT_DIR) / "raw" / f"{split}.jsonl"
    rej_path = Path(OUT_DIR) / "rejected" / f"{split}.jsonl"
    for p in [clean_path, raw_path, rej_path]:
        if p.exists(): p.unlink()

    created, attempts = 0, 0
    seen_hashes = set()

    while created < target_n:
        seeds = [build_single_seed() for _ in range(MAX_CONCURRENT)]
        tasks = [call_model_and_clean(f"{split}-{uuid.uuid4().hex[:12]}", s) for s in seeds]
        await asyncio.sleep(SLEEP_BETWEEN_BATCH)

        for coro in tasks:
            cleaned, raw, reason = await coro
            attempts += 1
            with open(raw_path, "a", encoding="utf-8") as fraw:
                fraw.write(json.dumps({"raw": raw, "reason": reason}, ensure_ascii=False)+"\n")
            if cleaned:
                h = short_hash({"in": cleaned["input"], "out": cleaned["output"]})
                if h in seen_hashes:
                    continue
                seen_hashes.add(h)
                with open(clean_path, "a", encoding="utf-8") as fcln:
                    fcln.write(json.dumps(cleaned, ensure_ascii=False)+"\n")
                created += 1
                if created % 60 == 0:
                    print(f"[{split}] {created}/{target_n} (attempts={attempts})")
            else:
                with open(rej_path, "a", encoding="utf-8") as frj:
                    frj.write(json.dumps({"reason": reason, "raw": raw}, ensure_ascii=False)+"\n")

    print(f"[{split}] Single-turn done: {created} items, attempts={attempts}")

async def producer_chain(split: str, num_chains: int, min_len=3, max_len=5):
    """Produce short multi-turn chains; each step saved as a standalone sample with chain_id."""
    out_chain = Path(OUT_DIR) / "clean" / f"{split}_chains.jsonl"
    raw_path = Path(OUT_DIR) / "raw" / f"{split}_chains.jsonl"
    rej_path = Path(OUT_DIR) / "rejected" / f"{split}_chains.jsonl"
    for p in [out_chain, raw_path, rej_path]:
        if p.exists(): p.unlink()

    created = 0
    while created < num_chains:
        domain, topic = random.choice(DOMAIN_TOPICS)
        grade = random.choice(GRADES)
        length = random.randint(min_len, max_len)

        # Start from canonical start_lesson
        status = random_status(domain, topic)
        status["lesson_stage"] = "not_in_lesson"
        status["current_lesson_title"] = topic
        status["current_lesson_progress"] = 0

        user_action = {"command":"start_lesson", "parameters":{"lesson_index":0}}
        chain_id = f"chain-{uuid.uuid4().hex[:10]}"

        for step in range(length):
            seed_turn = {
                "user_action": user_action,
                "status_dictionary": status,
                "seed_kind": f"chain_step_{step}",
                "domain": domain,
                "grade": grade,
                "topic": topic
            }
            cleaned, raw, reason = await call_model_and_clean(f"{split}-{chain_id}-{step}", seed_turn)

            with open(raw_path, "a", encoding="utf-8") as fraw:
                fraw.write(json.dumps({"raw": raw, "reason": reason}, ensure_ascii=False)+"\n")

            if not cleaned:
                with open(rej_path, "a", encoding="utf-8") as frj:
                    frj.write(json.dumps({"reason": reason, "raw": raw}, ensure_ascii=False)+"\n")
                break

            # Stamp chain id and write
            cleaned["meta"]["chain_id"] = chain_id
            with open(out_chain, "a", encoding="utf-8") as fcln:
                fcln.write(json.dumps(cleaned, ensure_ascii=False)+"\n")

            # Update status based on first update_status
            upd = extract_first_update_status(cleaned["output"]["actions"])
            if upd:
                status = apply_updates_to_status(status, upd)

            # Choose next user_action based on stage
            stage = status.get("lesson_stage","intuition")
            if stage == "intuition":
                user_action = {"command":"evaluate_paraphrase","parameters":{
                    "user_input": random.choice([
                        "Main idea is introduced via a story/analogy.", 
                        "Keep both sides equal; isolate variable carefully."
                    ])
                }}
            elif stage == "main_lesson":
                # Alternate checkpoint vs summary to teach routing
                user_action = random.choice([
                    {"command":"evaluate_answer","parameters":{"question_id":f"q_cp_{domain[:2]}_{topic[:6]}_{random.randint(1,99)}","user_answer":random.choice(["A","B","C","x=3","7"])}},
                    {"command":"evaluate_summary","parameters":{"user_input":"We covered the rule and example."}}
                ])
            elif stage == "practice":
                user_action = {"command":"evaluate_answer","parameters":{
                    "question_id": f"q_pr_{domain[:2]}_{topic[:6]}_{random.randint(1,99)}",
                    "user_answer": random.choice(["Correct","Incorrect","4","x=2"])
                }}
            elif stage == "lesson_complete":
                # Provide a next-lap choice
                user_action = random.choice([
                    {"command":"start_lesson","parameters":{"lesson_index":"next"}},
                    {"command":"show_status","parameters":{}}
                ])
                # chain end
                break
            else:
                break

        created += 1
        if created % 10 == 0:
            print(f"[{split}-chains] created {created}/{num_chains}")

    print(f"[{split}] Chains done: {created} chains")

def load_clean_all() -> List[dict]:
    all_clean = []
    for name in ["train","dev","test"]:
        p = Path(OUT_DIR) / "clean" / f"{name}.jsonl"
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                try: all_clean.append(json.loads(line))
                except: pass
        pc = Path(OUT_DIR) / "clean" / f"{name}_chains.jsonl"
        if pc.exists():
            for line in pc.read_text(encoding="utf-8").splitlines():
                try: all_clean.append(json.loads(line))
                except: pass
    return all_clean

def stratified_write(final: List[dict], split_name: str, n: int):
    random.shuffle(final)
    by_domain = {}
    for ex in final:
        d = ex.get("meta", {}).get("domain", "Other")
        by_domain.setdefault(d, []).append(ex)
    for d in by_domain:
        random.shuffle(by_domain[d])

    out, idx = [], {d:0 for d in by_domain}
    while len(out) < n and any(idx[d] < len(by_domain[d]) for d in by_domain):
        for d in list(by_domain.keys()):
            i = idx[d]
            if i < len(by_domain[d]) and len(out) < n:
                out.append(by_domain[d][i])
                idx[d] += 1
    path = Path(OUT_DIR) / f"{split_name}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for ex in out:
            f.write(json.dumps(ex, ensure_ascii=False)+"\n")
    print(f"Wrote {split_name} -> {len(out)}")

# --------------------------------
# MAIN
# --------------------------------
async def main():
    print(f"Collecting dataset with {MODEL_NAME}")
    print(f"Prompt: {PROMPT_PATH}")
    print(f"Out dir: {OUT_DIR}")

    train_single = int(TARGET_COUNTS["train"] * (1.0 - CHAIN_RATIO))
    dev_single   = int(TARGET_COUNTS["dev"] * (1.0 - CHAIN_RATIO))
    test_single  = int(TARGET_COUNTS["test"] * (1.0 - CHAIN_RATIO))

    train_chains = int(TARGET_COUNTS["train"] * CHAIN_RATIO)
    dev_chains   = int(TARGET_COUNTS["dev"] * CHAIN_RATIO)
    test_chains  = int(TARGET_COUNTS["test"] * CHAIN_RATIO)

    await producer_single("train", train_single)
    await producer_single("dev",   dev_single)
    await producer_single("test",  test_single)

    await producer_chain("train", train_chains)
    await producer_chain("dev",   dev_chains)
    await producer_chain("test",  test_chains)

    all_clean = load_clean_all()
    # Final stratified splits for balance
    stratified_write(all_clean, "train", TARGET_COUNTS["train"])
    stratified_write(all_clean, "dev",   TARGET_COUNTS["dev"])
    stratified_write(all_clean, "test",  TARGET_COUNTS["test"])
    print("Done.")

if __name__ == "__main__":
    asyncio.run(main())