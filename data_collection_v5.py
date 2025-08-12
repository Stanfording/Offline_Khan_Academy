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

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-1.5-flash-latest")
PROMPT_PATH = os.getenv("DELIGHT_PROMPT_PATH", "./prompt3_2.txt") # Assumes prompt is updated for init->course plan
OUT_DIR = os.getenv("OUT_DIR", "./mn_dataset_out_v9_menudriven")
SEED = int(os.getenv("SEED", "42"))
random.seed(SEED)

TARGET_COUNTS = {
    "train": int(os.getenv("TARGET_TRAIN", "6000")),
    "dev": int(os.getenv("TARGET_DEV", "200")),
    "test": int(os.getenv("TARGET_TEST", "200")),
}

CHAIN_RATIO = float(os.getenv("CHAIN_RATIO", "1.0"))
MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT", "8"))
SLEEP_BETWEEN_BATCH = float(os.getenv("SLEEP_BETWEEN_BATCH", "0.45"))

ENABLE_SECOND_PASS = os.getenv("ENABLE_SECOND_PASS", "true").lower() == "true"


#//-- NEW --//
# The single source of truth for course topics, based on the provided JSON.
COURSE_MENU_DATA = [
  { "category_id": "math_pre_k_8", "title": "Math: Pre-K - 8th grade", "icon": "🧮", "sub_topics": ["Pre-K through grade 2 (Khan Kids)", "2nd grade math", "3rd grade math", "4th grade math", "5th grade math", "6th grade math", "7th grade math", "8th grade math"] },
  { "category_id": "math_high_school_college", "title": "Math: high school & college", "icon": "🎓", "sub_topics": ["Algebra 1", "Geometry", "Algebra 2", "Integrated math 1", "Integrated math 2", "Integrated math 3", "Trigonometry", "Precalculus", "High school statistics", "AP®︎/College Calculus AB", "AP®︎/College Calculus BC", "Multivariable calculus", "Differential equations", "Linear algebra"] },
  { "category_id": "reading_language_arts", "title": "Reading & language arts", "icon": "📖", "sub_topics": ["Up to 2nd grade (Khan Kids) - reading_language_arts", "3rd grade - reading_language_arts", "4th grade reading and vocab", "5th grade reading and vocab", "6th grade reading and vocab", "Grammar"] },
  { "category_id": "science", "title": "Science", "icon": "🔬", "sub_topics": ["Middle school biology", "High school biology", "AP®︎/College Biology", "Cosmology and astronomy"] },
  { "category_id": "computing", "title": "Computing", "icon": "💻", "sub_topics": ["Intro to CS - Python", "Computer programming", "AP®︎/College Computer Science Principles", "Computers and the Internet"] },
  { "category_id": "life_skills", "title": "Life skills", "icon": "💡", "sub_topics": ["Financial literacy", "Internet safety", "Growth mindset", "College admissions", "Careers", "Personal finance"] }
]

# Create a flat list for easy random sampling. Each item is (category_title, sub_topic).
FLAT_COURSE_LIST: List[Tuple[str, str]] = []
for category in COURSE_MENU_DATA:
    for sub_topic in category["sub_topics"]:
        FLAT_COURSE_LIST.append((category["title"], sub_topic))

def random_course_topic() -> Tuple[str, str]:
    """Picks a random course from the defined menu."""
    return random.choice(FLAT_COURSE_LIST)


#//-- RETIRED --//
# The old system of domains, topics, and grades is now replaced by the COURSE_MENU_DATA.
# DOMAINS, TOPIC_GRADE_BANDS, valid_topics_for, and random_valid_domain_topic_grade are no longer needed.


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

BASE_SYSTEM_PROMPT = read_prompt()

# --------------------------------
# Helpers and minimal validators (no changes in this section)
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
                else: p = stage_defaults.get(stg, 10)
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
SEED_KINDS = [
    "init", "start_intuition", "paraphrase_good", "paraphrase_weak",
    "main_checkpoint_easy", "main_checkpoint_mid", "main_checkpoint_hard",
    "summary_good", "summary_weak",
    "practice_correct_easy", "practice_incorrect_easy",
    "practice_correct_mid", "practice_incorrect_mid",
    "practice_correct_hard", "practice_incorrect_hard",
    "lesson_complete", "handle_user_question", "give_feedback", "show_status"
]

#//-- MODIFIED --// Simplified function to generate a random status dict.
def random_status(sub_topic: str) -> Dict[str, Any]:
    return {
        "learning_confidence": random.randint(3, 8),
        "learning_interest": random.randint(4, 9),
        "learning_patience": random.randint(4, 9),
        "effort_focus": random.randint(5, 9),
        "weak_concept_spot": {},
        "current_lesson_progress": random.choice([0, 10, 40, 70, 90]),
        "current_lesson_title": sub_topic, # The title is the sub_topic
        "lesson_stage": random.choice(["not_in_lesson", "intuition", "main_lesson", "practice"])
    }

def build_single_seed() -> Dict[str, Any]:
    #//-- MODIFIED --// Use the new course sampler.
    category, sub_topic = random_course_topic()
    seed_kind = random.choice(SEED_KINDS)
    status = random_status(sub_topic)

    def qid(prefix: str, diff: str = ""):
        #//-- MODIFIED --// qid now uses category and sub_topic
        d = f"{prefix}_{diff}" if diff else prefix
        cat_abbr = "".join(filter(str.isupper, category)) or category[:3]
        return f"q_{cat_abbr.lower()}_{sub_topic[:6].lower().replace(' ','')}_{d}_{random.randint(1,999)}"

    if seed_kind == "init":
        ua = {
            "command": "init",
            "parameters": {
                "user_name": "Learner",
                "course_topic": sub_topic # Send the sub_topic directly
            }
        }
        status["lesson_stage"] = "not_in_lesson"
        status["current_lesson_progress"] = 0
    elif seed_kind == "start_intuition":
        ua = {"command":"start_lesson","parameters":{"lesson_index":0}}
        status["lesson_stage"] = "not_in_lesson"
        status["current_lesson_progress"] = 0
    elif seed_kind == "paraphrase_good":
        ua = {"command":"evaluate_paraphrase","parameters":{"user_input": "Main idea: balance the equation by doing the same to both sides."}}
    elif seed_kind == "paraphrase_weak":
        ua = {"command":"evaluate_paraphrase","parameters":{"user_input": "I’m not sure about the key idea."}}
    elif seed_kind in ["main_checkpoint_easy","main_checkpoint_mid","main_checkpoint_hard"]:
        diff = seed_kind.split("_")[-1]
        ua = {"command":"evaluate_answer","parameters":{"question_id": qid("checkpoint", diff), "user_answer": random.choice(["A","B","C","x=3","7"])}}
    elif seed_kind == "summary_good":
        ua = {"command":"start_practice_session","parameters":{"user_summary": "Clear steps, rule restated, example applied."}}
    elif seed_kind == "summary_weak":
        ua = {"command":"start_practice_session","parameters":{"user_summary": "I lost track of the steps."}}
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
    else: # Fallback for other commands
        ua = {"command": seed_kind, "parameters": {}}
        if seed_kind == "handle_user_question":
             ua["parameters"]["user_question_value"] = "How does this connect to real life?"

    stage_defaults = {
        "not_in_lesson": 0, "intuition": 10, "main_lesson": 40,
        "practice": 70, "lesson_complete": 100
    }
    stg = status.get("lesson_stage","intuition")
    if stg not in stage_defaults: stg = "intuition"
    p = stage_defaults.get(stg, 10)
    status["lesson_stage"] = stg
    status["current_lesson_progress"] = p

    #//-- MODIFIED --// The seed metadata is now category/sub_topic
    return {"user_action": ua, "status_dictionary": status, "seed_kind": seed_kind, "category": category, "sub_topic": sub_topic}

def build_messages(seed_turn: dict, strict_level: int = 1) -> List[Dict[str, Any]]:
    #//-- MODIFIED --// The context hint is now simpler and more direct.
    user_json = json.dumps({"user_action": seed_turn["user_action"], "status_dictionary": seed_turn["status_dictionary"]}, ensure_ascii=False)
    rubric = ("Teaching rubric: Use a friendly explanatory tone. Ask at least one self-question (why/how/what if) and then answer it succinctly before proceeding. Connect ideas progressively.")
    topic_hint = f"The user has selected the course '{seed_turn['sub_topic']}' from the '{seed_turn['category']}' category. Stay strictly on this topic."
    return [{"role":"user","parts":[
        BASE_SYSTEM_PROMPT, rubric, topic_hint, "Output strict JSON only: {\"actions\":[...]}",
        "Allowed commands only. Include update_status with lesson_stage/progress on stage changes.",
        "For an `init` command, you MUST generate a course plan. For a `start_lesson` command, you MUST start the first lesson step.",
        f"Next user turn:\n{user_json}"
    ]}]

# --------------------------------
# Gemini call + validation (no changes in this function)
# --------------------------------
async def call_model_and_clean(turn_id: str, seed_turn: dict) -> Tuple[Optional[dict], dict, str]:
    async def gen(strict_level: int):
        msgs = build_messages(seed_turn, strict_level=strict_level)
        try:
            resp = await model.generate_content_async(
                msgs,
                safety_settings={'HARM_CATEGORY_HARASSMENT':'BLOCK_NONE','HARM_CATEGORY_HATE_SPEECH':'BLOCK_NONE','HARM_CATEGORY_SEXUALLY_EXPLICIT':'BLOCK_NONE','HARM_CATEGORY_DANGEROUS_CONTENT':'BLOCK_NONE'},
                generation_config=gm_types.GenerationConfig(temperature=0.85 if strict_level==1 else 0.6, candidate_count=1)
            )
            return resp.text or ""
        except Exception as e: return {"__error__": str(e)}

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
            ok2, repaired2, _ = validate_actions_payload(raw_payload2)
            if ok2: repaired = repaired2; raw_payload = raw_payload2; ok = True
    if not ok: return None, raw_payload, f"Schema invalid: {reason}"

    cleaned = {
        "id": turn_id,
        "input": {"user_action": seed_turn["user_action"], "status_dictionary": seed_turn["status_dictionary"], "summary_context": f"Course: {seed_turn['sub_topic']} | Seed: {seed_turn['seed_kind']}"},
        "output": repaired,
        "meta": {"seed_kind": seed_turn["seed_kind"], "category": seed_turn["category"], "sub_topic": seed_turn["sub_topic"]}
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

async def producer_chain(split: str, num_chains: int, min_len=3, max_len=10):
    out_chain = Path(OUT_DIR) / "clean" / f"{split}_chains.jsonl"
    raw_path = Path(OUT_DIR) / "raw" / f"{split}_chains.jsonl"
    rej_path = Path(OUT_DIR) / "rejected" / f"{split}_chains.jsonl"
    for p in [out_chain, raw_path, rej_path]:
        if p.exists(): p.unlink()

    created = 0
    while created < num_chains:
        #//-- MODIFIED --// Use the new course sampler for chains
        category, sub_topic = random_course_topic()
        length = random.randint(min_len, max_len)
        chain_id = f"chain-{uuid.uuid4().hex[:10]}"

        status = random_status(sub_topic)
        status["lesson_stage"] = "not_in_lesson"
        status["current_lesson_progress"] = 0
        
        user_action = {"command": "init", "parameters": {"user_name": "Learner", "course_topic": sub_topic}}

        for step in range(length):
            seed_turn = {
                "user_action": user_action, "status_dictionary": status, "seed_kind": f"chain_step_{step}",
                "category": category, "sub_topic": sub_topic
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

            current_command = user_action.get("command")
            stage_now = status.get("lesson_stage", "not_in_lesson")

            if current_command == "init":
                user_action = {"command": "start_lesson", "parameters": {"lesson_index": 0}}
            elif stage_now == "intuition":
                user_action = {"command":"evaluate_paraphrase","parameters":{"user_input": "The analogy helped explain the core concept."}}
            elif stage_now == "main_lesson":
                 user_action = random.choice([
                    {"command":"evaluate_answer","parameters":{"question_id":f"q_checkpoint_{random.randint(1,99)}","user_answer":random.choice(["A","B","C","x=3","7"])}},
                    {"command":"start_practice_session","parameters":{"user_summary":"We covered the rule and saw an example of how it works."}}
                ])
            elif stage_now == "practice":
                user_action = {"command":"evaluate_answer","parameters":{"question_id": f"q_practice_{random.randint(1,99)}","user_answer": random.choice(["Correct","Incorrect","4","x=2"])}}
            elif stage_now == "lesson_complete":
                break
            else:
                break

        created += 1
        if created % 10 == 0:
            print(f"[{split}-chains] created {created}/{num_chains}")

    print(f"[{split}] Chains done: {created} chains")

# --------------------------------
# MAIN and Final Writing
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
    #//-- MODIFIED --// Stratify by 'category' instead of 'domain'
    random.shuffle(final)
    by_category: Dict[str, list] = {}
    for ex in final:
        cat = ex.get("meta", {}).get("category", "Other")
        by_category.setdefault(cat, []).append(ex)
    
    for cat in by_category:
        random.shuffle(by_category[cat])

    out, idx = [], {cat:0 for cat in by_category}
    while len(out) < n and any(idx[cat] < len(by_category[cat]) for cat in by_category):
        for cat in list(by_category.keys()):
            i = idx[cat]
            if i < len(by_category[cat]) and len(out) < n:
                out.append(by_category[cat][i])
                idx[cat] += 1
    
    path = Path(OUT_DIR) / f"{split_name}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for ex in out:
            f.write(json.dumps(ex, ensure_ascii=False)+"\n")
    print(f"Wrote {split_name} -> {len(out)}")

async def main():
    print(f"Collecting dataset with {MODEL_NAME}")
    print(f"Prompt: {PROMPT_PATH}")
    print(f"Out dir: {OUT_DIR}")
    print(f"Using menu-driven course topics.")

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
    print(f"Total cleaned examples generated: {len(all_clean)}")
    stratified_write(all_clean, "train", TARGET_COUNTS["train"])
    stratified_write(all_clean, "dev",   TARGET_COUNTS["dev"])
    stratified_write(all_clean, "test",  TARGET_COUNTS["test"])
    print("Done.")

if __name__ == "__main__":
    asyncio.run(main())