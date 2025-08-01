import os
import json
import logging
from typing import List, Dict, Any

# NEW: load .env
from dotenv import load_dotenv
load_dotenv()

from model_provider import ModelProvider, GeminiProvider, OllamaProvider

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MODEL_BACKEND = os.getenv("MODEL_BACKEND", "gemini").lower()
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
OLLAMA_ENDPOINT = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3n:e4b")
OLLAMA_PERSIST_DIR = os.getenv("OLLAMA_PERSIST_DIR", None)
OLLAMA_MAX_TURNS = int(os.getenv("OLLAMA_MAX_TURNS", "6"))
OLLAMA_SUMMARIZE_AFTER = int(os.getenv("OLLAMA_SUMMARIZE_AFTER", "8"))

PROMPT_PATH = os.path.join(os.path.dirname(__file__), "prompt3_1.txt")
with open(PROMPT_PATH, "r", encoding="utf-8") as f:
    base_prompt = f.read()

CONTRACT_APPENDIX = """
[OUTPUT CONTRACT — READ CAREFULLY]
- Always output STRICT JSON with a single top-level object: {"actions":[...]} and nothing else.
- Use only these commands in "actions":
  ui_display_notes, ui_short_answer, ui_mcq, ui_checkbox, ui_slider, ui_button, ui_rearrange_order, ui_drawing_board, update_status
- When lesson stage changes, always include an update_status with:
  - "lesson_stage": one of [ "not_in_lesson", "intuition", "main_lesson", "practice", "lesson_complete" ]
  - "current_lesson_progress": 10 for "intuition", 40 for "main_lesson", 70 for "practice", 100 for "lesson_complete".
- Maintain and update "current_lesson_title" appropriately.
- Persistent UI actions the frontend will call: show_status, skip_practice, give_feedback, start_lesson.
- evaluate_paraphrase for initial paraphrase; evaluate_summary for final summary.
Your response must be ONLY JSON. No extra commentary.
"""
TUTOR_SYSTEM_PROMPT = base_prompt.strip() + "\n" + CONTRACT_APPENDIX.strip()

def build_provider() -> ModelProvider:
    if MODEL_BACKEND == "ollama":
        logging.info(f"Using OllamaProvider ({OLLAMA_MODEL} @ {OLLAMA_ENDPOINT})")
        return OllamaProvider(
            endpoint=OLLAMA_ENDPOINT,
            model=OLLAMA_MODEL,
            system_prompt=TUTOR_SYSTEM_PROMPT,
            max_window_turns=OLLAMA_MAX_TURNS,
            summarize_after_turns=OLLAMA_SUMMARIZE_AFTER,
            persist_dir=OLLAMA_PERSIST_DIR
        )
    logging.info(f"Using GeminiProvider ({GEMINI_MODEL})")
    return GeminiProvider(api_key=GEMINI_API_KEY, model=GEMINI_MODEL)

provider: ModelProvider = build_provider()

NUM_RECENT_CHATS_TO_KEEP = 3
NUM_CHATS_BEFORE_SUMMARIZATION = 6

def _extract_balanced_json_block(s: str):
    start = s.find("{")
    if start == -1:
        return None
    stack = 0
    for i in range(start, len(s)):
        if s[i] == "{":
            stack += 1
        elif s[i] == "}":
            stack -= 1
            if stack == 0:
                return s[start:i+1]
    return None

def _summarize_conversation(messages_to_summarize):
    extracted_content = []
    for msg in messages_to_summarize:
        if msg["role"] == "user":
            try:
                user_data = json.loads(msg["content"])
                if "user_action" in user_data and "parameters" in user_data["user_action"]:
                    action_params = user_data["user_action"]["parameters"]
                    command = user_data["user_action"]["command"]
                    if command == "generate_course_plan" and "query" in action_params:
                        extracted_content.append(f"User wants: {action_params['query']}")
                    elif command in ["evaluate_paraphrase", "evaluate_summary"] and "user_input" in action_params:
                        extracted_content.append(f"User wrote: {action_params['user_input']}")
                    elif command == "evaluate_answer":
                        extracted_content.append(f"Answered '{action_params.get('question_id','?')}': {action_params.get('user_answer')}")
                    elif command in ["show_status", "give_feedback", "start_lesson", "skip_practice"]:
                        extracted_content.append(f"User clicked '{command}' params {json.dumps(action_params)}")
                elif "status_dictionary" in user_data:
                    extracted_content.append(f"State: {json.dumps(user_data['status_dictionary'])}")
            except json.JSONDecodeError:
                extracted_content.append(f"User: {msg['content']}")
        elif msg["role"] == "assistant":
            try:
                assistant = json.loads(msg["content"])
                for action in assistant.get("actions", []):
                    if action["command"] == "ui_display_notes":
                        extracted_content.append(f"Note: {action['parameters']['content']}")
                    elif "question_text" in action.get("parameters", {}):
                        extracted_content.append(f"Q: {action['parameters']['question_text']}")
                    elif action["command"] == "update_status":
                        ups = action["parameters"].get("updates", {})
                        if "lesson_stage" in ups:
                            extracted_content.append(f"Stage -> {ups['lesson_stage']}")
                        if "current_lesson_progress" in ups:
                            extracted_content.append(f"Progress -> {ups['current_lesson_progress']}")
            except json.JSONDecodeError:
                extracted_content.append(f"Assistant: {msg['content']}")
    text = "\n".join(extracted_content).strip()
    if not text:
        return "No relevant previous conversation to summarize."
    try:
        return provider.summarize("Summarize key facts for context:\n" + text)
    except Exception as e:
        logging.error(f"Summarization error via provider {provider.name()}: {e}")
        return "Previous conversation context lost due to summarization error."

def ask_tutor(user_action_data, current_status_dictionary, messages_history=None, session_id: str = None):
    logging.info(f"-> ask_tutor using {provider.name()} cmd={user_action_data.get('command')} sid={session_id}")
    if messages_history is None or len(messages_history) == 0 or not (messages_history[0].get("role") == "system" and messages_history[0].get("content") == TUTOR_SYSTEM_PROMPT):
        messages_history = [{"role": "system", "content": TUTOR_SYSTEM_PROMPT}]

    user_input_content = json.dumps({
        "user_action": user_action_data,
        "status_dictionary": current_status_dictionary
    })
    messages_history.append({"role": "user", "content": user_input_content})

    current_payload_messages_for_model: List[Dict[str, Any]] = []
    conversational_messages = messages_history[1:]

    if (len(conversational_messages)) > (2 * NUM_CHATS_BEFORE_SUMMARIZATION):
        num_messages_to_keep_recent = (2 * NUM_RECENT_CHATS_TO_KEEP) + 1
        num_messages_to_summarize = len(conversational_messages) - num_messages_to_keep_recent
        if num_messages_to_summarize > 0:
            summary_text = _summarize_conversation(conversational_messages[:num_messages_to_summarize])
            current_payload_messages_for_model.append({"role": "user", "parts": [f"Context summary: {summary_text}"]})
            current_payload_messages_for_model.append({"role": "model", "parts": ["Summary received."]})
        for msg in conversational_messages[num_messages_to_summarize:]:
            role = "model" if msg["role"] == "assistant" else "user"
            current_payload_messages_for_model.append({"role": role, "parts": [msg["content"]]})
    else:
        for msg in conversational_messages:
            role = "model" if msg["role"] == "assistant" else "user"
            current_payload_messages_for_model.append({"role": role, "parts": [msg["content"]]})

    if not current_payload_messages_for_model:
        current_payload_messages_for_model = [{"role": "user", "parts": [user_input_content]}]

    first_parts = current_payload_messages_for_model[0]["parts"]
    final_messages = [
        {"role": "user", "parts": [TUTOR_SYSTEM_PROMPT] + first_parts},
    ] + current_payload_messages_for_model[1:] + [
        {"role": "user", "parts": ['Output must be strict JSON only: {"actions":[...]}']}
    ]

    try:
        raw_text = provider.generate_content(final_messages, temperature=0.7, session_id=session_id)
        try:
            llm_response_json = json.loads(raw_text)
        except json.JSONDecodeError:
            candidate = _extract_balanced_json_block(raw_text)
            if candidate:
                llm_response_json = json.loads(candidate)
            else:
                return {"actions": [{"command": "ui_display_notes", "parameters": {"content": "Mr. Delight had a parsing hiccup. Please try again."}}]}, messages_history

        messages_history.append({"role": "assistant", "content": json.dumps(llm_response_json)})
        return llm_response_json, messages_history

    except Exception as e:
        logging.error(f"Model call failed ({provider.name()}): {e}")
        return {"actions": [{"command": "ui_display_notes", "parameters": {"content": f"Mr. Delight is offline: {str(e)}. Please refresh and try again."}}]}, messages_history