import os
import json
import time
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

# --------------------------------
# CONFIG
# --------------------------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError("Set GOOGLE_API_KEY in your environment or .env")

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
PROMPT_PATH = os.getenv("DELIGHT_PROMPT_PATH", "./prompt3_2.txt")
OUT_DIR = os.getenv("OUT_DIR", "./mn_dataset_out")
SEED = int(os.getenv("SEED", "42"))
random.seed(SEED)

# Total counts (adjust quickly)
TARGET_COUNTS = {
    "train": 1800,   # was 1200
    "dev": 240,      # was 150
    "test": 240,     # was 150
}

# Rate/concurrency
MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT", "8"))
SLEEP_BETWEEN_BATCH = float(os.getenv("SLEEP_BETWEEN_BATCH", "0.45"))

# --------------------------------
# WIDE SUBJECT/TOPIC LIBRARY
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

# Populate a flat selection list for random sampling with domain info
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
# Gemini
# --------------------------------
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel(MODEL_NAME)

# --------------------------------
# Prompt
# --------------------------------
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
    t = text.lower()
    return any(k in t for k in ["solve", "equation", "factor", "simplify", "x =", "y =", "derivative", "integral"])

def validate_actions_payload(payload: dict) -> Tuple[bool, dict, str]:
    if not isinstance(payload, dict) or "actions" not in payload or not isinstance(payload["actions"], list):
        return False, payload, "Missing actions"

    repaired = []
    for a in payload["actions"]:
        if not isinstance(a, dict) or "command" not in a: return False, payload, "Bad action item"
        cmd = a["command"]
        if cmd not in ALLOWED_COMMANDS: return False, payload, f"Disallowed command: {cmd}"
        params = a.get("parameters", {})
        if not isinstance(params, dict): params = {}

        # math keypad hinting
        if cmd == "ui_short_answer":
            qtext = params.get("question_text","")
            if is_equation_like(qtext):
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

# --------------------------------
# Seed synthesis with greater variety
# --------------------------------
SEED_KINDS = [
    "init","plan","start_intuition","paraphrase_good","paraphrase_weak",
    "main_checkpoint_easy","main_checkpoint_mid","main_checkpoint_hard",
    "summary_good","summary_weak",
    "practice_correct_easy","practice_incorrect_easy",
    "practice_correct_mid","practice_incorrect_mid",
    "practice_correct_hard","practice_incorrect_hard",
    "lesson_complete","handle_user_question","skip_practice","give_feedback","show_status"
]

def random_status(domain: str, topic: str) -> Dict[str, Any]:
    base = {
        "learning_confidence": random.randint(3,8),
        "learning_interest": random.randint(4,9),
        "learning_patience": random.randint(4,9),
        "effort_focus": random.randint(5,9),
        "weak_concept_spot": {},
        "current_lesson_progress": random.choice([0,10,40,60,70,90]),
        "current_lesson_title": f"{topic}",
        "lesson_stage": random.choice(["not_in_lesson","intuition","main_lesson","practice"])
    }
    # add weak spots contextually
    if domain == "Math":
        base["weak_concept_spot"] = {random.choice(["Fractions","BalanceMoves","QuadraticForm","PrimeNumbers","Slopes"]): random.randint(3,7)}
    elif domain == "Physics":
        base["weak_concept_spot"] = {random.choice(["Forces","Kinematics","Circuits"]): random.randint(3,7)}
    elif domain == "Chemistry":
        base["weak_concept_spot"] = {random.choice(["Stoichiometry","AcidsBases","Bonds"]): random.randint(3,7)}
    elif domain == "Biology":
        base["weak_concept_spot"] = {random.choice(["Photosynthesis","Genetics","Ecosystems"]): random.randint(3,7)}
    return base

def build_seed() -> Dict[str, Any]:
    domain, topic = random.choice(DOMAIN_TOPICS)
    grade = random.choice(GRADES)
    seed_kind = random.choice(SEED_KINDS)

    status = random_status(domain, topic)

    def qid(prefix: str, diff: str = ""):
        d = f"{prefix}_{diff}" if diff else prefix
        return f"q_{domain[:2].lower()}_{topic[:6].lower()}_{d}_{random.randint(1,999)}"

    # Construct user_action reflecting the seed_kind
    if seed_kind == "init":
        ua = {"command":"init","parameters":{"user_name":"Learner"}}
    elif seed_kind == "plan":
        ua = {"command":"generate_course_plan","parameters":{"query":f"{domain}: {topic} for {grade}"}}
    elif seed_kind == "start_intuition":
        ua = {"command":"start_lesson","parameters":{"lesson_index":0}}
    elif seed_kind == "paraphrase_good":
        ua = {"command":"evaluate_paraphrase","parameters":{"user_input":"You keep the core rule in mind and restate the main idea clearly."}}
    elif seed_kind == "paraphrase_weak":
        ua = {"command":"evaluate_paraphrase","parameters":{"user_input":"I’m confused about the main point."}}
    elif seed_kind in ["main_checkpoint_easy","main_checkpoint_mid","main_checkpoint_hard"]:
        ua = {"command":"evaluate_answer","parameters":{"question_id": qid("checkpoint", seed_kind.split("_")[-1]), "user_answer": random.choice(["A","B","C","7","x=4"])}}
    elif seed_kind == "summary_good":
        ua = {"command":"evaluate_summary","parameters":{"user_input":"Key steps and reasons summarized accurately with examples."}}
    elif seed_kind == "summary_weak":
        ua = {"command":"evaluate_summary","parameters":{"user_input":"Not sure about some of the steps."}}
    elif seed_kind.startswith("practice_correct"):
        ua = {"command":"evaluate_answer","parameters":{"question_id": qid("lesson", seed_kind.split("_")[-1]), "user_answer": random.choice(["Correct","4","x=3","Area=12"]) }}
    elif seed_kind.startswith("practice_incorrect"):
        ua = {"command":"evaluate_answer","parameters":{"question_id": qid("lesson", seed_kind.split("_")[-1]), "user_answer": random.choice(["Incorrect","9","x=10","Area=5"]) }}
    elif seed_kind == "lesson_complete":
        ua = {"command":"skip_practice","parameters":{}}
    elif seed_kind == "handle_user_question":
        ua = {"command":"handle_user_question","parameters":{"user_question_value":"How does this connect to real life?"}}
    elif seed_kind == "give_feedback":
        ua = {"command":"give_feedback","parameters":{}}
    elif seed_kind == "show_status":
        ua = {"command":"show_status","parameters":{}}
    else:
        ua = {"command":"init","parameters":{"user_name":"Learner"}}

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
            f"Next user turn:\n{user_json}"
        ]}
    ]

# --------------------------------
# Generate + Validate
# --------------------------------
async def generate_one(turn_id: str, seed_turn: dict) -> Tuple[Optional[dict], dict, str]:
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
                temperature=0.8,  # slightly higher for variety
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

async def producer(split: str, target_n: int):
    clean_path = Path(OUT_DIR) / "clean" / f"{split}.jsonl"
    raw_path = Path(OUT_DIR) / "raw" / f"{split}.jsonl"
    rej_path = Path(OUT_DIR) / "rejected" / f"{split}.jsonl"
    for p in [clean_path, raw_path, rej_path]:
        if p.exists(): p.unlink()

    created, attempts = 0, 0
    seen_hashes = set()

    while created < target_n:
        seeds = [build_seed() for _ in range(MAX_CONCURRENT)]
        tasks = [generate_one(f"{split}-{uuid.uuid4().hex[:12]}", s) for s in seeds]
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

    print(f"[{split}] Done: {created} items, attempts={attempts}")

def load_clean_splits() -> List[dict]:
    all_clean = []
    for split in ["train","dev","test"]:
        f = Path(OUT_DIR) / "clean" / f"{split}.jsonl"
        if f.exists():
            for line in f.read_text(encoding="utf-8").splitlines():
                try:
                    all_clean.append(json.loads(line))
                except:
                    pass
    return all_clean

def stratified_write(final: List[dict], split_name: str, n: int):
    # Keep domain/topic diversity
    random.shuffle(final)
    by_domain = {}
    for ex in final:
        d = ex.get("meta", {}).get("domain", "Other")
        by_domain.setdefault(d, []).append(ex)
    # round-robin picking per domain
    buckets = [(d, lst) for d, lst in by_domain.items()]
    for _, lst in buckets: random.shuffle(lst)

    out = []
    idx_by_domain = {d:0 for d,_ in buckets}
    while len(out) < n and any(idx_by_domain[d] < len(lst) for d,lst in buckets):
        for d, lst in buckets:
            i = idx_by_domain[d]
            if i < len(lst) and len(out) < n:
                out.append(lst[i])
                idx_by_domain[d] += 1
    out_path = Path(OUT_DIR) / f"{split_name}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in out:
            f.write(json.dumps(ex, ensure_ascii=False)+"\n")
    print(f"Wrote {split_name} -> {len(out)}")

async def main():
    print(f"Generating with {MODEL_NAME}")
    print(f"Prompt: {PROMPT_PATH}")
    print(f"Out: {OUT_DIR}")

    await producer("train", TARGET_COUNTS["train"])
    await producer("dev", TARGET_COUNTS["dev"])
    await producer("test", TARGET_COUNTS["test"])

    all_clean = load_clean_splits()
    # Split again with stratification to ensure domain balance
    stratified_write(all_clean, "train", TARGET_COUNTS["train"])
    stratified_write(all_clean, "dev", TARGET_COUNTS["dev"])
    stratified_write(all_clean, "test", TARGET_COUNTS["test"])
    print("All done.")

if __name__ == "__main__":
    asyncio.run(main())