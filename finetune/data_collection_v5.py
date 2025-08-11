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

#//-- MODIFIED --// Using a newer, potentially more capable model if available
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-1.5-flash-latest")
PROMPT_PATH = os.getenv("DELIGHT_PROMPT_PATH", "./prompt3_2.txt") # Make sure this file has the updated prompt
OUT_DIR = os.getenv("OUT_DIR", "./mn_dataset_out_v8_courseplan")
SEED = int(os.getenv("SEED", "42"))
random.seed(SEED)

TARGET_COUNTS = {
    "train": int(os.getenv("TARGET_TRAIN", "1")),
    "dev": int(os.getenv("TARGET_DEV", "200")),
    "test": int(os.getenv("TARGET_TEST", "200")),
}

CHAIN_RATIO = float(os.getenv("CHAIN_RATIO", "1.0"))
MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT", "8"))
SLEEP_BETWEEN_BATCH = float(os.getenv("SLEEP_BETWEEN_BATCH", "0.45"))

ENABLE_SECOND_PASS = os.getenv("ENABLE_SECOND_PASS", "true").lower() == "true"

# --------------------------------
# SUBJECTS/TOPICS, Grade Bands (kept only for sampling)
# (No changes in this section)
# --------------------------------
GRADES = [
    "K", "1st", "2nd", "3rd", "4th", "5th",
    "6th", "7th", "8th", "9th", "10th", "11th", "12th",
    "Intro College"
]

def grade_band(g: str) -> str:
    if g in ["K","1st","2nd"]: return "K-2"
    if g in ["3rd","4th","5th"]: return "3-5"
    if g in ["6th","7th","8th"]: return "6-8"
    if g in ["9th","10th","11th","12th"]: return "9-12"
    return "College"

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

TOPIC_GRADE_BANDS = {
  "Math": {
    "K-2": ["Counting and Place Value","Addition/Subtraction Basics"],
    "3-5": ["Multiplication/Division Basics","Fractions & Decimals","Geometry Basics","Area/Perimeter","Statistics & Data"],
    "6-8": ["Ratios & Proportions","Percentages & Interest","Probability Basics","Slope & Graphing Lines","Linear Equations","Inequalities","Volume & Surface Area"],
    "9-12": ["Systems of Equations","Quadratics & Factoring","Polynomials","Exponents & Radicals","Trigonometry (Intro)"],
    "College": ["Limits (Intro)","Derivatives (Intro)","Integrals (Intro)"]
  },
  "Physics": {
    "6-8": ["Units & Measurement","Motion (1D/2D)","Forces & Newton's Laws","Energy & Work","Waves & Sound"],
    "9-12": ["Momentum & Collisions","Circular Motion & Gravitation","Light & Optics","Electricity (Basics)","Circuits (DC)","Magnetism (Intro)"],
    "College": []
  },
  "Chemistry": {
    "6-8": ["Matter & States","Periodic Table"],
    "9-12": ["Atomic Structure","Chemical Bonds","Chemical Reactions","Stoichiometry","Acids & Bases","Gases","Thermochemistry","Solutions & Concentration","Equilibrium (Intro)"],
    "College": []
  },
  "Biology": {
    "3-5": ["Ecosystems & Food Webs"],
    "6-8": ["Cell Structure","Photosynthesis","Human Body Systems","Ecosystems & Food Webs"],
    "9-12": ["Cellular Respiration","DNA/RNA & Protein Synthesis","Genetics (Basics)","Evolution & Natural Selection","Biodiversity & Conservation"],
    "College": []
  },
  "Computer Science": {
    "6-8": ["Binary & Hex","Boolean Logic","Basic Python","Control Flow","Debugging Strategies","Functions & Parameters"],
    "9-12": ["Algorithms (Sorting & Searching)","Data Structures (Arrays/Lists/Stacks/Queues)","Big-O (Intro)","Recursion (Intro)","Basic JavaScript"],
    "College": []
  },
  "Language Arts": {
    "K-2": ["Phonics & Sight Words","Reading Comprehension"],
    "3-5": ["Main Idea & Details","Summarizing","Grammar & Parts of Speech","Punctuation","Narrative Writing"],
    "6-8": ["Inference","Essay Structure","Persuasive Writing","Poetry (Intro)"],
    "9-12": [],
    "College": []
  },
  "History": {
    "3-5": ["Ancient Civilizations"],
    "6-8": ["Middle Ages","Renaissance","American Revolution","Industrial Revolution","World Wars Overview"],
    "9-12": ["US Constitution (Basics)","Civil Rights Movement","Modern Globalization (Intro)"],
    "College": []
  },
  "Finance": {
    "3-5": ["Counting Money","Savings vs. Spending","Budgeting Basics"],
    "6-8": ["Simple Interest","Taxes (Basics)","Credit & Debt (Intro)"],
    "9-12": ["Compound Interest (Intro)"],
    "College": []
  },
  "Art & Music": {
    "3-5": ["Color Wheel & Mixing","Drawing Basics","Rhythm & Beat"],
    "6-8": ["Perspective (Intro)","Scales (Intro)","Reading Music (Intro)"],
    "9-12": [],
    "College": []
  },
  "Study Skills": {
    "3-5": ["Note-Taking Strategies"],
    "6-8": ["Memory Techniques (Spaced Repetition)","Test-Taking Strategies","Goal Setting & Planning"],
    "9-12": ["Test-Taking Strategies","Goal Setting & Planning"],
    "College": []
  },
  "Maker/Practical": {
    "3-5": ["Cooking Measurements","Maps & Coordinates"],
    "6-8": ["Unit Conversions","Intro Electronics (Circuits)","Environmental Science Projects (Water Quality)"],
    "9-12": ["3D Printing Basics"],
    "College": []
  },
  "Test Prep": {
    "9-12": ["SAT Math (Linear/Quadratic)","SAT Reading (Passage Strategy)","SAT Writing (Grammar)","ACT Science (Data & Experiments)"],
    "College": []
  }
}

def valid_topics_for(domain: str, grade: str) -> List[str]:
    gb = grade_band(grade)
    table = TOPIC_GRADE_BANDS.get(domain, {})
    bands_priority = ["K-2","3-5","6-8","9-12","College"]
    if gb not in bands_priority:
        return []
    idx = bands_priority.index(gb)
    allowed = []
    for i in range(idx+1):
        allowed.extend(table.get(bands_priority[i], []))
    if not allowed:
        allowed = DOMAINS.get(domain, [])
    return allowed

DOMAIN_TOPICS: List[Tuple[str,str]] = []
for d, topics in DOMAINS.items():
    for t in topics:
        DOMAIN_TOPICS.append((d, t))

# --------------------------------
# IO
# (No changes in this section)
# --------------------------------
out_dir = Path(OUT_DIR)
(out_dir / "raw").mkdir(parents=True, exist_ok=True)
(out_dir / "clean").mkdir(exist_ok=True)
(out_dir / "rejected").mkdir(exist_ok=True)

# --------------------------------
# Gemini init
# (No changes in this section)
# --------------------------------
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel(MODEL_NAME)

def read_prompt() -> str:
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read().strip()

BASE_SYSTEM_PROMPT = read_prompt()

# --------------------------------
# Helpers and minimal validators
# (No changes in this section)
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
        if cmd == "update_status":
            ups = params.get("updates", {})
            if isinstance(ups, dict):
                stg = ups.get("lesson_stage")
                prog = ups.get("current_lesson_progress", -1)
                stage_defaults = {
                    "not_in_lesson": 0, "intuition": 10, "main_lesson": 40,
                    "practice": 70, "lesson_complete": 100
                }
                if stg not in stage_defaults: stg = "intuition"
                try: p = int(prog)
                except: p = stage_defaults[stg]
                if stg == "practice" and p >= 70 and p < 100: p = max(70, min(90, p))
                else: p = stage_defaults[stg]
                ups["lesson_stage"] = stg
                ups["current_lesson_progress"] = str(p)
                params["updates"] = ups
        repaired.append({"command": cmd, "parameters": params})
    if not repaired: return False, payload, "Empty actions"
    return True, {"actions": repaired}, ""

def apply_updates_to_status(status: dict, action: dict) -> dict:
    ups = action.get("parameters",{}).get("updates",{})
    new_status = json.loads(json.dumps(status))
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
    for f in ["learning_confidence","learning_interest","learning_patience","effort_focus"]:
        if f in new_status and isinstance(new_status[f], (int,float)):
            new_status[f] = max(1, min(10, new_status[f]))
    stage_defaults = {
        "not_in_lesson": 0, "intuition": 10, "main_lesson": 40,
        "practice": 70, "lesson_complete": 100
    }
    stg = new_status.get("lesson_stage", "intuition")
    prog = new_status.get("current_lesson_progress", 10)
    if stg not in stage_defaults: stg = "intuition"
    if stg == "practice":
        if not isinstance(prog, int):
            try: prog = int(prog)
            except: prog = 70
        prog = max(70, min(90, prog))
    else:
        prog = stage_defaults.get(stg, 10)
    new_status["lesson_stage"] = stg
    new_status["current_lesson_progress"] = prog
    if not new_status.get("current_lesson_title"):
        new_status["current_lesson_title"] = status.get("current_lesson_title","")
    return new_status

def extract_first_update_status(actions: List[dict]) -> Optional[dict]:
    for a in actions:
        if a.get("command") == "update_status":
            return a
    return None

# --------------------------------
# Seed builders
# --------------------------------
#//-- MODIFIED --// `plan` is removed as it's merged into `init`.
SEED_KINDS = [
    "init", "start_intuition", "paraphrase_good", "paraphrase_weak",
    "main_checkpoint_easy", "main_checkpoint_mid", "main_checkpoint_hard",
    "summary_good", "summary_weak",
    "practice_correct_easy", "practice_incorrect_easy",
    "practice_correct_mid", "practice_incorrect_mid",
    "practice_correct_hard", "practice_incorrect_hard",
    "lesson_complete", "handle_user_question", "give_feedback", "show_status"
]

def random_valid_domain_topic_grade() -> Tuple[str,str,str]:
    for _ in range(200):
        domain = random.choice(list(DOMAINS.keys()))
        grade = random.choice(GRADES)
        allowed = valid_topics_for(domain, grade)
        if allowed:
            topic = random.choice(allowed)
            return domain, topic, grade
    raise RuntimeError("Could not sample a valid domain/topic/grade after 200 attempts.")

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
    return base

def build_single_seed() -> Dict[str, Any]:
    domain, topic, grade = random_valid_domain_topic_grade()
    seed_kind = random.choice(SEED_KINDS)
    status = random_status(domain, topic)

    def qid(prefix: str, diff: str = ""):
        d = f"{prefix}_{diff}" if diff else prefix
        return f"q_{domain[:2].lower()}_{topic[:6].lower()}_{d}_{random.randint(1,999)}"

    #//-- MODIFIED --// 'init' now creates a course plan directly. 'plan' is removed.
    if seed_kind == "init":
        # The new 'init' command takes a topic and produces a course plan.
        ua = {
            "command": "init",
            "parameters": {
                "user_name": "Learner",
                "course_topic": f"{topic} for {grade} students"
            }
        }
        status["lesson_stage"] = "not_in_lesson"
        status["current_lesson_progress"] = 0
    #//-- MODIFIED --// 'plan' seed kind is removed.
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
            "How does this connect to real life?", "Why do we divide here instead of subtract?",
            "What if the data is noisy?", "How do we know this rule applies here?"
        ])}}
    elif seed_kind == "give_feedback":
        ua = {"command":"give_feedback","parameters":{}}
    elif seed_kind == "show_status":
        ua = {"command":"show_status","parameters":{}}
    else: # Fallback, should not be common
        ua = {"command": "init", "parameters": {"user_name": "Learner", "course_topic": f"{topic} for {grade}"}}
        status["lesson_stage"] = "not_in_lesson"
        status["current_lesson_progress"] = 0

    stage_defaults = {
        "not_in_lesson": 0, "intuition": 10, "main_lesson": 40,
        "practice": 70, "lesson_complete": 100
    }
    stg = status.get("lesson_stage","intuition")
    if stg not in stage_defaults: stg = "intuition"
    if stg == "practice":
        p = status.get("current_lesson_progress", 70)
        if not isinstance(p, int):
            try: p = int(p)
            except: p = 70
        p = max(70, min(90, p))
    else:
        p = stage_defaults.get(stg, 10)
    status["lesson_stage"] = stg
    status["current_lesson_progress"] = p

    return {"user_action": ua, "status_dictionary": status, "seed_kind": seed_kind, "domain": domain, "grade": grade, "topic": topic}

def build_messages(seed_turn: dict, strict_level: int = 1) -> List[Dict[str, Any]]:
    # (No changes in this function)
    user_json = json.dumps({"user_action": seed_turn["user_action"], "status_dictionary": seed_turn["status_dictionary"]}, ensure_ascii=False)
    rubric = ("Teaching rubric: Use a friendly explanatory tone. Ask at least one self-question (why/how/what if) and then answer it succinctly before proceeding. Connect ideas progressively.")
    domain_hint = f"Stay strictly on topic for {seed_turn['domain']} / {seed_turn['topic']} (grade {seed_turn['grade']}). Do not drift to other domains."
    return [{"role":"user","parts":[
        BASE_SYSTEM_PROMPT, rubric, domain_hint, "Output strict JSON only: {\"actions\":[...]}",
        "Allowed commands only. Include update_status with lesson_stage/progress on stage changes.",
        "Use hints.math_input only for equation-like short answers.",
        "Route inputs: initial_paraphrase→evaluate_paraphrase, final_summary→start_practice_session, q_*→evaluate_answer.",
        "Maintain realistic grade-topic pairing and avoid off-grade complexity.",
        "Ensure status updates are consistent, monotonic progression through stages.",
        "Include at least one rhetorical or self-question in display notes.",
        f"Next user turn:\n{user_json}"
    ]}]

# --------------------------------
# Gemini call + minimal validation (optional second pass)
# (No changes in this function)
# --------------------------------
async def call_model_and_clean(turn_id: str, seed_turn: dict) -> Tuple[Optional[dict], dict, str]:
    async def gen(strict_level: int):
        msgs = build_messages(seed_turn, strict_level=strict_level)
        try:
            resp = await model.generate_content_async(
                msgs,
                safety_settings={
                    'HARM_CATEGORY_HARASSMENT':'BLOCK_NONE','HARM_CATEGORY_HATE_SPEECH':'BLOCK_NONE',
                    'HARM_CATEGORY_SEXUALLY_EXPLICIT':'BLOCK_NONE','HARM_CATEGORY_DANGEROUS_CONTENT':'BLOCK_NONE'
                },
                generation_config=gm_types.GenerationConfig(temperature=0.85 if strict_level==1 else 0.6, candidate_count=1)
            )
            return resp.text or ""
        except Exception as e:
            return {"__error__": str(e)}

    raw_text = await gen(strict_level=1)
    if isinstance(raw_text, dict) and "__error__" in raw_text:
        return None, {"error": raw_text["__error__"]}, f"API error: {raw_text['__error__']}"

    raw_payload = ensure_json(raw_text)
    if not raw_payload and ENABLE_SECOND_PASS:
        raw_text2 = await gen(strict_level=2)
        raw_payload = ensure_json(raw_text2)
    if not raw_payload: return None, {"raw": raw_text}, "Non-JSON"

    ok, repaired, reason = validate_actions_payload(raw_payload)
    if not ok and ENABLE_SECOND_PASS:
        raw_text2 = await gen(strict_level=2)
        raw_payload2 = ensure_json(raw_text2)
        if raw_payload2:
            ok2, repaired2, reason2 = validate_actions_payload(raw_payload2)
            if ok2:
                repaired = repaired2; raw_payload = raw_payload2; ok = True
    if not ok: return None, raw_payload, f"Schema invalid: {reason}"

    cleaned = {
        "id": turn_id,
        "input": {"user_action": seed_turn["user_action"], "status_dictionary": seed_turn["status_dictionary"],
                  "summary_context": f"{seed_turn['domain']} / {seed_turn['topic']} for {seed_turn['grade']} | Seed: {seed_turn['seed_kind']}"},
        "output": repaired,
        "meta": {"seed_kind": seed_turn["seed_kind"], "domain": seed_turn["domain"], "grade": seed_turn["grade"], "topic": seed_turn["topic"]}
    }
    return cleaned, raw_payload, ""

# --------------------------------
# Producers
# --------------------------------
async def producer_single(split: str, target_n: int):
    # (No changes in this function)
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
        for coro in asyncio.as_completed(tasks):
            cleaned, raw, reason = await coro
            attempts += 1
            with open(raw_path, "a", encoding="utf-8") as fraw: fraw.write(json.dumps({"raw": raw, "reason": reason}, ensure_ascii=False)+"\n")
            if cleaned:
                h = short_hash({"in": cleaned["input"], "out": cleaned["output"]})
                if h in seen_hashes: continue
                seen_hashes.add(h)
                with open(clean_path, "a", encoding="utf-8") as fcln: fcln.write(json.dumps(cleaned, ensure_ascii=False)+"\n")
                created += 1
                if created % 60 == 0: print(f"[{split}] {created}/{target_n} (attempts={attempts})")
            else:
                with open(rej_path, "a", encoding="utf-8") as frj: frj.write(json.dumps({"reason": reason, "raw": raw}, ensure_ascii=False)+"\n")
    print(f"[{split}] Single-turn done: {created} items, attempts={attempts}")

async def producer_chain(split: str, num_chains: int, min_len=3, max_len=6):
    out_chain = Path(OUT_DIR) / "clean" / f"{split}_chains.jsonl"
    raw_path = Path(OUT_DIR) / "raw" / f"{split}_chains.jsonl"
    rej_path = Path(OUT_DIR) / "rejected" / f"{split}_chains.jsonl"
    for p in [out_chain, raw_path, rej_path]:
        if p.exists(): p.unlink()

    created = 0
    while created < num_chains:
        domain, topic, grade = random_valid_domain_topic_grade()
        length = random.randint(min_len, max_len)
        chain_id = f"chain-{uuid.uuid4().hex[:10]}"

        #//-- MODIFIED --// Chain now starts from the very beginning.
        # Initial status is always 'not_in_lesson'
        status = random_status(domain, topic)
        status["lesson_stage"] = "not_in_lesson"
        status["current_lesson_title"] = topic
        status["current_lesson_progress"] = 0
        
        # The first action is now `init` which generates the course plan.
        user_action = {
            "command": "init",
            "parameters": {
                "user_name": "Learner",
                "generate_course_plan": f"{topic} for {grade}"
            }
        }

        for step in range(length):
            seed_turn = {
                "user_action": user_action, "status_dictionary": status, "seed_kind": f"chain_step_{step}",
                "domain": domain, "grade": grade, "topic": topic
            }
            cleaned, raw, reason = await call_model_and_clean(f"{split}-{chain_id}-{step}", seed_turn)

            with open(raw_path, "a", encoding="utf-8") as fraw: fraw.write(json.dumps({"raw": raw, "reason": reason}, ensure_ascii=False)+"\n")

            if not cleaned:
                with open(rej_path, "a", encoding="utf-8") as frj: frj.write(json.dumps({"reason": reason or 'turn-reject', "raw": raw}, ensure_ascii=False)+"\n")
                break

            cleaned["meta"]["chain_id"] = chain_id
            upd = extract_first_update_status(cleaned["output"]["actions"])
            if upd: status = apply_updates_to_status(status, upd)

            with open(out_chain, "a", encoding="utf-8") as fcln: fcln.write(json.dumps(cleaned, ensure_ascii=False)+"\n")

            #//-- MODIFIED --// Logic to decide the next action in the chain.
            current_command = user_action.get("command")
            stage_now = status.get("lesson_stage", "not_in_lesson")

            if current_command == "init":
                # After the course plan is generated, the only logical next step is to start lesson 0.
                user_action = {"command": "start_lesson", "parameters": {"lesson_index": 0}}
            elif stage_now == "intuition":
                user_action = {"command":"evaluate_paraphrase","parameters":{"user_input": "The analogy helped explain the core concept."}}
            elif stage_now == "main_lesson":
                 user_action = random.choice([
                    {"command":"evaluate_answer","parameters":{"question_id":f"q_cp_{domain[:2]}_{topic[:6]}_{random.randint(1,99)}","user_answer":random.choice(["A","B","C","x=3","7"])}},
                    {"command":"start_practice_session","parameters":{"user_summary":"We covered the rule and saw an example of how it works."}}
                ])
            elif stage_now == "practice":
                user_action = {"command":"evaluate_answer","parameters":{"question_id": f"q_pr_{domain[:2]}_{topic[:6]}_{random.randint(1,99)}","user_answer": random.choice(["Correct","Incorrect","4","x=2"])}}
            elif stage_now == "lesson_complete":
                user_action = {"command":"start_lesson","parameters":{"lesson_index":"next"}}
                break # End this chain, as the lesson is complete
            else:
                break # Break on any unexpected state

        created += 1
        if created % 10 == 0:
            print(f"[{split}-chains] created {created}/{num_chains}")

    print(f"[{split}] Chains done: {created} chains")

# --------------------------------
# MAIN and Final Writing
# (No changes in these functions)
# --------------------------------
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
    by_domain: Dict[str, list] = {}
    for ex in final:
        d = ex.get("meta", {}).get("domain", "Other")
        by_domain.setdefault(d, []).append(ex)
    for d in by_domain: random.shuffle(by_domain[d])
    out, idx = [], {d:0 for d in by_domain}
    while len(out) < n and any(idx[d] < len(by_domain[d]) for d in by_domain):
        for d in list(by_domain.keys()):
            i = idx[d]
            if i < len(by_domain[d]) and len(out) < n:
                out.append(by_domain[d][i])
                idx[d] += 1
    path = Path(OUT_DIR) / f"{split_name}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for ex in out: f.write(json.dumps(ex, ensure_ascii=False)+"\n")
    print(f"Wrote {split_name} -> {len(out)}")

async def main():
    print(f"Collecting dataset with {MODEL_NAME}")
    print(f"Prompt: {PROMPT_PATH}")
    print(f"Out dir: {OUT_DIR}")
    print(f"ENABLE_SECOND_PASS={ENABLE_SECOND_PASS}")

    train_single = int(TARGET_COUNTS["train"] * (1.0 - CHAIN_RATIO))
    dev_single   = int(TARGET_COUNTS["dev"] * (1.0 - CHAIN_RATIO))
    test_single  = int(TARGET_COUNTS["test"] * (1.0 - CHAIN_RATIO))

    train_chains = int(TARGET_COUNTS["train"] * CHAIN_RATIO)
    dev_chains   = int(TARGET_COUNTS["dev"] * CHAIN_RATIO)
    test_chains  = int(TARGET_COUNTS["test"] * CHAIN_RATIO)

    tasks = []
    if train_single > 0: tasks.append(producer_single("train", train_single))
    if dev_single > 0: tasks.append(producer_single("dev",   dev_single))
    if test_single > 0: tasks.append(producer_single("test",  test_single))
    if train_chains > 0: tasks.append(producer_chain("train", train_chains))
    if dev_chains > 0: tasks.append(producer_chain("dev",   dev_chains))
    if test_chains > 0: tasks.append(producer_chain("test",  test_chains))

    await asyncio.gather(*tasks)

    all_clean = load_clean_all()
    stratified_write(all_clean, "train", TARGET_COUNTS["train"])
    stratified_write(all_clean, "dev",   TARGET_COUNTS["dev"])
    stratified_write(all_clean, "test",  TARGET_COUNTS["test"])
    print("Done.")

if __name__ == "__main__":
    # In some environments (like Jupyter), you need to get/create an event loop
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    loop.run_until_complete(main())