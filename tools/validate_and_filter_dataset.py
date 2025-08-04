import os
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import argparse

# Reuse your constraints (keep in sync with collector)
GRADES = ["K","1st","2nd","3rd","4th","5th","6th","7th","8th","9th","10th","11th","12th","Intro College"]

def grade_band(g: str) -> str:
    if g in ["K","1st","2nd"]: return "K-2"
    if g in ["3rd","4th","5th"]: return "3-5"
    if g in ["6th","7th","8th"]: return "6-8"
    if g in ["9th","10th","11th","12th"]: return "9-12"
    return "College"

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

SELF_Q_TRIGGERS = ["why", "how", "what if", "could we", "does that mean", "so then", "?"]

def valid_topic(domain: str, topic: str, grade: str) -> bool:
    gb = grade_band(grade)
    allowed = []
    order = ["K-2","3-5","6-8","9-12","College"]
    if gb not in order:
        return False
    idx = order.index(gb)
    m = TOPIC_GRADE_BANDS.get(domain, {})
    for i in range(idx+1):
        allowed.extend(m.get(order[i], []))
    return topic in allowed

def contains_self_q(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in SELF_Q_TRIGGERS)

def first_update(actions: List[dict]) -> Optional[dict]:
    for a in actions:
        if a.get("command") == "update_status":
            return a
    return None

def normalize_stage_progress(stage: str, progress: Any) -> Tuple[str, int]:
    stage_defaults = {
        "not_in_lesson": 0,
        "intuition": 10,
        "main_lesson": 40,
        "practice": 70,
        "lesson_complete": 100
    }
    if stage not in stage_defaults:
        stage = "intuition"
    try:
        p = int(progress)
    except:
        p = stage_defaults[stage]
    # Clamp to canonical; allow practice 70..100
    if stage == "practice":
        p = max(70, min(100, p))
    else:
        p = stage_defaults[stage]
    return stage, p

def validate_turn(ex: dict) -> Tuple[bool, str]:
    meta = ex.get("meta", {})
    domain = meta.get("domain","")
    topic = meta.get("topic","")
    grade = meta.get("grade","")
    if not valid_topic(domain, topic, grade):
        return False, f"grade-topic-invalid: {domain}/{topic} for {grade}"

    # Self-questioning: require in at least one ui_display_notes
    actions = ex.get("output",{}).get("actions",[])
    saw_selfq = False
    for a in actions:
        if a.get("command") == "ui_display_notes":
            if contains_self_q(a.get("parameters",{}).get("content","")):
                saw_selfq = True
                break
    if not saw_selfq:
        return False, "no-self-questioning"
    return True, ""

def pass_chain_consistency(turns: List[dict]) -> Tuple[List[dict], List[dict]]:
    """Ensure monotonic stage/progress and topical consistency for a chain."""
    if not turns:
        return [], []
    domain = turns[0]["meta"]["domain"]
    topic = turns[0]["meta"]["topic"]
    ok, rej = [], []
    prev_stage = "not_in_lesson"
    prev_prog = 0
    order = ["not_in_lesson","intuition","main_lesson","practice","lesson_complete"]
    for ex in turns:
        # Topic drift
        if ex["meta"]["domain"] != domain or ex["meta"]["topic"] != topic:
            rej.append((ex, "topic-drift"))
            continue

        # Pull update_status for stage/progress
        actions = ex.get("output",{}).get("actions",[])
        upd = first_update(actions)
        if upd:
            ups = upd.get("parameters",{}).get("updates",{})
            stage = ups.get("lesson_stage", ex["input"]["status_dictionary"].get("lesson_stage","intuition"))
            prog = ups.get("current_lesson_progress", ex["input"]["status_dictionary"].get("current_lesson_progress",10))
        else:
            stage = ex["input"]["status_dictionary"].get("lesson_stage","intuition")
            prog = ex["input"]["status_dictionary"].get("current_lesson_progress",10)

        stage, prog = normalize_stage_progress(stage, prog)

        # Monotonic rule: stage must not go backwards, progress must not decrease unless stage advances canonically from 0->10->40->70->100
        try:
            ps = order.index(prev_stage)
            cs = order.index(stage)
        except ValueError:
            rej.append((ex,"bad-stage"))
            continue

        if cs < ps:
            rej.append((ex,"stage-regression"))
            continue
        if cs == ps and prog < prev_prog:
            rej.append((ex,"progress-regression"))
            continue

        # Write back normalized stage/progress to input status to keep consistency
        ex["input"]["status_dictionary"]["lesson_stage"] = stage
        ex["input"]["status_dictionary"]["current_lesson_progress"] = prog

        ok_flag, reason = validate_turn(ex)
        if ok_flag:
            ok.append(ex)
            prev_stage, prev_prog = stage, prog
        else:
            rej.append((ex, reason))
    return ok, rej

def read_jsonl(path: Path) -> List[dict]:
    if not path.exists(): return []
    out=[]
    for line in path.read_text(encoding="utf-8").splitlines():
        line=line.strip()
        if not line: continue
        try: out.append(json.loads(line))
        except: pass
    return out

def write_jsonl(path: Path, items: List[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path,"w",encoding="utf-8") as f:
        for ex in items:
            f.write(json.dumps(ex, ensure_ascii=False)+"\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="collector OUT_DIR")
    ap.add_argument("--out_dir", required=True, help="filtered OUT_DIR")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = ["train","dev","test"]
    kept_total = 0
    rejected_report = []

    for split in splits:
        # plain
        plain = read_jsonl(in_dir / f"{split}.jsonl")
        kept_plain, rej_plain = [], []
        for ex in plain:
            ok, reason = validate_turn(ex)
            if ok:
                kept_plain.append(ex)
            else:
                rej_plain.append((ex, reason))

        # chains
        chain_items = read_jsonl(in_dir / f"{split}_chains.jsonl")
        # group by chain_id
        by_chain: Dict[str, List[dict]] = {}
        for ex in chain_items:
            cid = ex.get("meta",{}).get("chain_id","")
            if cid: by_chain.setdefault(cid,[]).append(ex)
        kept_chain = []
        for cid, steps in by_chain.items():
            steps_sorted = sorted(steps, key=lambda x: x["id"])
            ok_chain, rej_chain = pass_chain_consistency(steps_sorted)
            kept_chain.extend(ok_chain)
            for _, r in rej_chain:
                rejected_report.append({"split": split, "chain": cid, "reason": r})

        # write
        write_jsonl(out_dir / f"{split}.jsonl", kept_plain)
        write_jsonl(out_dir / f"{split}_chains.jsonl", kept_chain)
        kept_total += len(kept_plain) + len(kept_chain)

    # write merged final like the collector did
    merged = []
    for split in splits:
        merged.extend(read_jsonl(out_dir / f"{split}.jsonl"))
        merged.extend(read_jsonl(out_dir / f"{split}_chains.jsonl"))
    # Stratified shuffle not strictly needed here; you can reuse your function if desired
    write_jsonl(out_dir / "train.jsonl", [ex for ex in merged if 'train' in ex['id']][:])
    write_jsonl(out_dir / "dev.jsonl",   [ex for ex in merged if 'dev' in ex['id']][:])
    write_jsonl(out_dir / "test.jsonl",  [ex for ex in merged if 'test' in ex['id']][:])

    with open(out_dir / "rejected_report.json", "w", encoding="utf-8") as f:
        json.dump(rejected_report, f, ensure_ascii=False, indent=2)

    print(f"Kept total items: {kept_total}")
    print(f"Rejected count: {len(rejected_report)}. See rejected_report.json")

if __name__ == "__main__":
    main()