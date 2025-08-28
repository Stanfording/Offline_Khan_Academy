import os
import json
import logging
from typing import List, Dict, Any

# NEW: load .env
from dotenv import load_dotenv
load_dotenv()

# MODIFIED: Import NvidiaProvider
from model_provider import ModelProvider, GeminiProvider, OllamaProvider, NvidiaProvider

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MODEL_BACKEND = os.getenv("MODEL_BACKEND", "ollama").lower()
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash-latest")
OLLAMA_ENDPOINT = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma2:9b")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
NVIDIA_MODEL = os.getenv("NVIDIA_MODEL", "openai/gpt-oss-20b")

PROMPT_PATH = os.path.join(os.path.dirname(__file__), "prompt3_2.txt" )
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
# This is no longer the full system prompt, just the user-level message history start.
# The provider will now construct the real system/developer prompts.
TUTOR_HISTORY_START = [{"role": "system", "content": "Conversation Start"}]


def build_provider() -> ModelProvider:
    if MODEL_BACKEND == "ollama":
        logging.info(f"Using OllamaProvider ({OLLAMA_MODEL} @ {OLLAMA_ENDPOINT})")
        # For Ollama, we still combine the prompt.
        full_prompt = base_prompt.strip() + "\n" + CONTRACT_APPENDIX.strip()
        # This part of OllamaProvider would need to be updated if it used a stateful approach.
        # Since we made it stateless, we'll handle this in ask_tutor.
        return OllamaProvider(endpoint=OLLAMA_ENDPOINT, model=OLLAMA_MODEL)

    elif MODEL_BACKEND == "nvidia":
        logging.info(f"Using NvidiaProvider ({NVIDIA_MODEL})")
        key = os.getenv("NVIDIA_API_KEY")
        if not key: logging.error("NVIDIA_API_KEY environment variable not found!")
        else: logging.info(f"NVIDIA_API_KEY loaded successfully. Starts with: {key[:4]}...")
        
        # **THE FIX**: Pass the separated prompt components to the provider.
        return NvidiaProvider(
            api_key=NVIDIA_API_KEY,
            model=NVIDIA_MODEL,
            developer_prompt=base_prompt,
            output_contract=CONTRACT_APPENDIX
        )
    
    logging.info(f"Using GeminiProvider ({GEMINI_MODEL})")
    return GeminiProvider(api_key=GEMINI_API_KEY, model=GEMINI_MODEL)

provider: ModelProvider = build_provider()

def _extract_image_from_action(user_action: Dict[str, Any]) -> Dict[str, Any] | None:
    try:
        params = (user_action or {}).get("parameters", {})
        if "user_answer" in params and isinstance(params["user_answer"], dict):
            img = params["user_answer"]
            if "image_b64" in img:
                return {"b64": img.get("image_b64", ""), "mime_type": img.get("mime_type", "image/png")}
    except Exception: pass
    return None

def _extract_balanced_json_block(s: str):
    start = s.find("{");
    if start == -1: return None
    stack = 0
    for i in range(start, len(s)):
        if s[i] == "{": stack += 1
        elif s[i] == "}":
            stack -= 1
            if stack == 0: return s[start:i+1]
    return None

def ask_tutor(user_action_data, current_status_dictionary, messages_history: List[Dict[str, str]]=None, session_id: str = None):
    logging.info(f"-> ask_tutor using {provider.name()} cmd={user_action_data.get('command')} sid={session_id}")

    # For Ollama/Gemini, prepend the full prompt if history is new.
    # For Nvidia, this system message will be ignored and replaced by the structured prompt.
    if messages_history is None or not messages_history:
        full_prompt_for_others = base_prompt.strip() + "\n" + CONTRACT_APPENDIX.strip()
        messages_history = [{"role": "system", "content": full_prompt_for_others}]

    user_input_content = json.dumps({"user_action": user_action_data, "status_dictionary": current_status_dictionary}, ensure_ascii=False)
    current_user_message: Dict[str, Any] = {"role": "user", "content": user_input_content}
    image_data = _extract_image_from_action(user_action_data)
    if image_data: current_user_message["image"] = image_data
    messages_history.append(current_user_message)

    try:
        raw_text_response = provider.generate_content(messages=messages_history, temperature=1.0, session_id=session_id)

        logging.info(f"raw_text_response, {raw_text_response}")
        
        llm_response_json = None
        try:
            llm_response_json = json.loads(raw_text_response)
        except json.JSONDecodeError:
            candidate = _extract_balanced_json_block(raw_text_response)
            if candidate:
                try: llm_response_json = json.loads(candidate)
                except json.JSONDecodeError: pass
            if not llm_response_json:
                logging.error(f"Failed to parse JSON from LLM response: {raw_text_response}")
                return {"actions": [{"command": "ui_display_notes", "parameters": {"content": "Mr. Delight had a parsing hiccup. Please try again."}}]}, messages_history

        logging.info(f"<- ask_tutor received valid JSON from {provider.name()} for cmd={user_action_data.get('command')}")
        messages_history.append({"role": "assistant", "content": json.dumps(llm_response_json, ensure_ascii=False)})
        return llm_response_json, messages_history

    except Exception as e:
        logging.error(f"Model call failed ({provider.name()}): {e}", exc_info=True)
        return {"actions": [{"command": "ui_display_notes", "parameters": {"content": f"Mr. Delight is offline: {str(e)}. Please refresh and try again."}}]}, messages_history