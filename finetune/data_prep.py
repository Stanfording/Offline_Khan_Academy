import json
from typing import List, Dict, Any, Tuple

def load_chain_jsonl(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                continue
    return items

def to_chat_sample(ex: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert one chain step into a chat-like sample for SFT:
    - system: distilled Mr. Delight prompt
    - user: includes user_action + status_dictionary (shortened)
    - assistant: the model JSON output (actions only)
    """
    user_action = ex["input"]["user_action"]
    status = ex["input"]["status_dictionary"]
    actions = ex["output"]["actions"]

    # Thin status for token economy
    status_small = {
        "lesson_stage": status.get("lesson_stage"),
        "current_lesson_title": status.get("current_lesson_title"),
        "current_lesson_progress": status.get("current_lesson_progress"),
        "learning_confidence": status.get("learning_confidence"),
        "learning_interest": status.get("learning_interest"),
        "learning_patience": status.get("learning_patience"),
        "effort_focus": status.get("effort_focus"),
    }

    user_msg = {
        "user_action": user_action,
        "status_dictionary": status_small
    }
    assistant_msg = {"actions": actions}

    return {
        "messages": [
            {"role": "system", "content": open("distilled_prompt.txt","r",encoding="utf-8").read()},
            {"role": "user", "content": json.dumps(user_msg, ensure_ascii=False)},
            {"role": "assistant", "content": json.dumps(assistant_msg, ensure_ascii=False)}
        ]
    }

def build_chat_dataset(train_path: str, dev_path: str = None) -> Tuple[List[dict], List[dict]]:
    train = load_chain_jsonl(train_path)
    if dev_path:
        dev = load_chain_jsonl(dev_path)
    else:
        dev = []
    train_chats = [to_chat_sample(x) for x in train]
    dev_chats = [to_chat_sample(x) for x in dev]
    return train_chats, dev_chats