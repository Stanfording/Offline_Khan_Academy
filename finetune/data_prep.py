import json
import re
from typing import List, Dict, Any, Tuple
from pathlib import Path
from collections import defaultdict, Counter


def _safe_read_text(path: str) -> str:
    """Safely read text from a file, returning empty string on failure."""
    try:
        return Path(path).read_text(encoding="utf-8")
    except Exception:
        print(f"Warning: Could not read system prompt file at {path}. Using empty prompt.")
        return ""


def load_chain_jsonl(path: str) -> List[Dict[str, Any]]:
    """
    Loads a JSONL file where each line is a dictionary. Skips invalid lines.
    """
    items: List[Dict[str, Any]] = []
    p = Path(path)
    if not p.exists():
        print(f"Warning: Data file not found at {path}. Returning empty list.")
        return items

    with p.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
                if isinstance(obj, dict):
                    items.append(obj)
            except json.JSONDecodeError:
                # Skip malformed line
                continue
    return items


def _thin_status(status: Dict[str, Any]) -> Dict[str, Any]:
    """
    Keep only key status fields to reduce token usage.
    """
    if not isinstance(status, dict):
        return {}
    keys_to_keep = [
        "lesson_stage",
        "current_lesson_title",
        "current_lesson_progress",
        "learning_confidence",
        "learning_interest",
        "learning_patience",
        "effort_focus",
    ]
    return {k: status[k] if k in status else None for k in keys_to_keep if k in status}


def _step_index_from_id(sample_id: str) -> int:
    """
    Parse step index from an id like 'train-chain-<hash>-<step>'.
    Returns 0 if not parseable.
    """
    try:
        return int(str(sample_id).rsplit("-", 1)[-1])
    except Exception:
        return 0


_CHAIN_ID_RE = re.compile(r".*?chain-([^-]+)-\d+$")


def _chain_key(ex: Dict[str, Any]) -> str:
    """
    Determine a stable chain key:
    1) Use meta.chain_id if present and non-empty.
    2) Else, parse the stable hash from id pattern '...chain-<hash>-<step>'.
    3) Else, 'unknown_chain'.
    """
    meta = ex.get("meta") or {}
    if isinstance(meta, dict):
        cid = meta.get("chain_id")
        if isinstance(cid, str) and cid:
            return cid

    sid = ex.get("id", "")
    m = _CHAIN_ID_RE.match(str(sid))
    if m:
        return "chain-" + m.group(1)

    return "unknown_chain"


def _group_by_chain(items: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Groups steps by chain key and sorts each chain by step index.
    """
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for ex in items:
        groups[_chain_key(ex)].append(ex)
    for cid in groups:
        groups[cid].sort(key=lambda e: _step_index_from_id(e.get("id", "")))
    return groups


def _step_to_pair(step: Dict[str, Any]) -> Tuple[str, str]:
    """
    Converts one raw step into a (user_content_str, assistant_content_str) pair.
    Both are JSON strings. Returns ('','') if cannot form a valid pair.
    """
    if not isinstance(step, dict):
        return "", ""

    input_obj = step.get("input", {}) or {}
    output_obj = step.get("output", {}) or {}

    if not isinstance(input_obj, dict) or not isinstance(output_obj, dict):
        return "", ""

    user_action = input_obj.get("user_action", {}) or {}
    status = input_obj.get("status_dictionary", {}) or {}
    actions = output_obj.get("actions", []) or []

    # We require at least one action to constitute an assistant reply
    if not isinstance(actions, list) or len(actions) == 0:
        return "", ""

    user_msg_content = json.dumps(
        {"user_action": user_action if isinstance(user_action, dict) else {},
         "status_dictionary": _thin_status(status)},
        ensure_ascii=False,
    )
    assistant_msg_content = json.dumps({"actions": actions}, ensure_ascii=False)

    return user_msg_content, assistant_msg_content


def chain_to_incremental_chats(
    chain_steps: List[Dict[str, Any]],
    system_prompt: str,
    keep_last_k: int | None = None,  # if set, only keep the last K incremental samples
) -> List[Dict[str, Any]]:
    """
    Convert a chain's steps into multiple incremental conversation samples:
      sample_1: [system, user1, assistant1]
      sample_2: [system, user1, assistant1, user2, assistant2]
      ...
      sample_K: [system, ..., userK, assistantK]
    If keep_last_k is provided, only keep the last K samples to control dataset size.
    """
    # Build clean turns
    turns: List[Tuple[str, str]] = []
    for step in chain_steps:
        u, a = _step_to_pair(step)
        if u and a:
            turns.append((u, a))

    if not turns:
        return []

    # Build incremental samples
    samples: List[Dict[str, Any]] = []
    for i in range(1, len(turns) + 1):
        msgs = [{"role": "system", "content": system_prompt}]
        for j in range(i):
            msgs.append({"role": "user", "content": turns[j][0]})
            msgs.append({"role": "assistant", "content": turns[j][1]})
        samples.append({"messages": msgs})

    if keep_last_k is not None and keep_last_k > 0 and len(samples) > keep_last_k:
        samples = samples[-keep_last_k:]

    return samples


def build_chat_dataset(
    train_path: str,
    dev_path: str | None = None,
    system_prompt_path: str = "distilled_prompt.txt",
    keep_last_k: int | None = None,
    verbose_stats: bool = True,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Build dataset where each chain is expanded into multiple incremental samples.
    keep_last_k: keep only the last K incremental samples per chain (optional).
    """
    print("Building chat dataset (incremental multi-turn samples per chain)...")
    system_prompt = _safe_read_text(system_prompt_path)

    train_items = load_chain_jsonl(train_path)
    dev_items = load_chain_jsonl(dev_path) if dev_path else []

    if verbose_stats:
        print(f"Train raw lines: {len(train_items)}; Dev raw lines: {len(dev_items)}")

    train_groups = _group_by_chain(train_items)
    dev_groups = _group_by_chain(dev_items)

    # Optional stats on chains
    if verbose_stats:
        tr_counts = Counter({cid: len(steps) for cid, steps in train_groups.items()})
        print(f"Unique train chains: {len(tr_counts)}")
        if tr_counts:
            one = sum(1 for v in tr_counts.values() if v == 1)
            ge2 = sum(1 for v in tr_counts.values() if v >= 2)
            print(f"Train chains with 1 step: {one}; with >=2 steps: {ge2}")
            print("Top 5 train chains by steps:", tr_counts.most_common(5))

    # Expand to incremental samples
    train_chats: List[Dict[str, Any]] = []
    for steps in train_groups.values():
        train_chats.extend(chain_to_incremental_chats(steps, system_prompt, keep_last_k=keep_last_k))

    dev_chats: List[Dict[str, Any]] = []
    for steps in dev_groups.values():
        dev_chats.extend(chain_to_incremental_chats(steps, system_prompt, keep_last_k=keep_last_k))

    print(f"Built {len(train_chats)} training samples and {len(dev_chats)} dev samples.")
    return train_chats, dev_chats