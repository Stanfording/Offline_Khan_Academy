import os
import json
import logging
from typing import List, Dict, Any, Optional
import requests

logger = logging.getLogger(__name__)

try:
    import google.generativeai as genai
    from google.generativeai import types as gemini_types
except Exception:
    genai = None
    gemini_types = None

class ModelProvider:
    def generate_content(self, messages: List[Dict[str, Any]], temperature: float = 0.7, session_id: Optional[str] = None) -> str:
        raise NotImplementedError
    def summarize(self, prompt: str, temperature: float = 0.2) -> str:
        raise NotImplementedError
    def name(self) -> str:
        raise NotImplementedError

class GeminiProvider(ModelProvider):
    def __init__(self, api_key: str, model: str = "gemini-2.5-flash-lite"):
        if genai is None:
            raise RuntimeError("google-generativeai not installed.")
        if not api_key:
            raise ValueError("Gemini API key is required for GeminiProvider.")
        genai.configure(api_key=api_key)
        self.model_name = model
        self.main_model = genai.GenerativeModel(self.model_name)
        self.summarizer_model = genai.GenerativeModel(self.model_name)

    def generate_content(self, messages: List[Dict[str, Any]], temperature: float = 0.7, session_id: Optional[str] = None) -> str:
        response = self.main_model.generate_content(
            messages,
            safety_settings={
                'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE',
                'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE',
                'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE',
                'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE'
            },
            generation_config=gemini_types.GenerationConfig(
                temperature=temperature,
                candidate_count=1,
            ),
        )
        return response.text or ""

    def summarize(self, prompt: str, temperature: float = 0.2) -> str:
        response = self.summarizer_model.generate_content(
            [{"role": "user", "parts": [prompt]}],
            safety_settings={
                'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE',
                'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE',
                'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE',
                'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE'
            },
            generation_config=gemini_types.GenerationConfig(
                temperature=temperature,
                candidate_count=1,
            ),
        )
        return response.text or ""

    def name(self) -> str:
        return f"gemini:{self.model_name}"

class OllamaProvider(ModelProvider):
    """
    Efficient chat memory for Ollama via:
      - KV cache reuse (context)
      - Rolling short window of turns
      - Compact memory summary
      - System prompt seeded once into KV cache, then a tiny system anchor each turn
    """
    def __init__(
        self,
        endpoint: str = "http://localhost:11434",
        model: str = "gemma3n:e4b",
        system_prompt: Optional[str] = None,
        max_window_turns: int = 6,        # number of recent (user/assistant) messages to keep
        summarize_after_turns: int = 8,   # when exceeded, build memory_summary
        persist_dir: Optional[str] = None
    ):
        self.endpoint = endpoint.rstrip("/")
        self.chat_url = f"{self.endpoint}/api/chat"
        self.model = model
        self.long_system_prompt = (system_prompt or "").strip()
        self.max_window_turns = max_window_turns
        self.summarize_after_turns = summarize_after_turns
        self.persist_dir = persist_dir

        # Per-session state
        # {
        #   session_id: {
        #     "context": [...],
        #     "short_messages": [ {role, content} ... ],  # last 2*max_window_turns entries
        #     "memory_summary": str or "",
        #     "system_seeded": bool
        #   }
        # }
        self.sessions: Dict[str, Dict[str, Any]] = {}

        if self.persist_dir:
            os.makedirs(self.persist_dir, exist_ok=True)

    def name(self) -> str:
        return f"ollama:{self.model}@{self.endpoint}"

    def _get_session(self, session_id: Optional[str]) -> Dict[str, Any]:
        sid = session_id or "__default__"
        sess = self.sessions.get(sid)
        if not sess:
            sess = {
                "context": None,
                "short_messages": [],
                "memory_summary": "",
                "system_seeded": False
            }
            self.sessions[sid] = sess
        return sess

    def _save_session(self, session_id: Optional[str]):
        if not self.persist_dir or not session_id:
            return
        try:
            path = os.path.join(self.persist_dir, f"{session_id}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.sessions.get(session_id, {}), f, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to persist Ollama session {session_id}: {e}")

    def _post_chat(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        r = requests.post(self.chat_url, json=payload, timeout=120)
        r.raise_for_status()
        return r.json()

    def _maybe_summarize(self, sess: Dict[str, Any]):
        # If too many short messages, summarize them into memory_summary
        if len(sess["short_messages"]) > 2 * self.summarize_after_turns:
            # Build a compact plain text transcript
            chunks = []
            for m in sess["short_messages"]:
                prefix = "U:" if m["role"] == "user" else "A:"
                chunks.append(f"{prefix} {m['content']}")
            transcript = "\n".join(chunks)
            prompt = (
                "Summarize the following conversation into a compact memory "
                "useful for continuing the lesson. Preserve key facts, the current "
                "lesson title, stage, weak spots, and goals. Keep it under 150 tokens.\n\n"
                + transcript
            )
            # Use the same model for summarization to avoid extra dependencies
            try:
                payload = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": "You are a concise memory builder."},
                        {"role": "user", "content": prompt}
                    ],
                    "stream": False,
                    "options": { "temperature": 0.2 }
                }
                resp = self._post_chat(payload)
                summary = (resp.get("message", {}) or {}).get("content", "") or ""
                if summary:
                    sess["memory_summary"] = summary.strip()
                    # Keep only the last few messages for recency
                    sess["short_messages"] = sess["short_messages"][-2*self.max_window_turns:]
            except Exception as e:
                logger.warning(f"Ollama summarization failed: {e}")

    def generate_content(self, messages: List[Dict[str, Any]], temperature: float = 0.7, session_id: Optional[str] = None) -> str:
        sess = self._get_session(session_id)

        # Convert incoming (Gemini-format) messages to plain role/content pairs (only current turn)
        # We do not include previous turns here; we maintain them in sess["short_messages"].
        current_turn_msgs = []
        for m in messages:
            role = m.get("role", "user")
            content = "\n".join([str(p) for p in m.get("parts", []) if isinstance(p, str)]).strip()
            if not content:
                continue
            # Map Gemini 'model' role to 'assistant'
            role = "assistant" if role == "model" else "user"
            current_turn_msgs.append({"role": role, "content": content})

        # Detect special image marker on the last user message and extract base64

        images_b64 = []
        for m in current_turn_msgs:
            if m["role"] == "user" and m["content"]:
                try:
                    lines = m["content"].splitlines()
                    if lines:
                        maybe = lines[-1].strip()
                        obj = json.loads(maybe)
                        if isinstance(obj, dict) and "__image__" in obj:
                            im = obj["__image__"]
                            b64 = im.get("b64", "")
                            if b64:
                                images_b64.append(b64)
                                # Remove the marker line from the actual text the model will see
                                m["content"] = "\n".join(lines[:-1]).rstrip()
                except Exception:
                    pass

        # Build the chat payload efficiently:
        ollama_messages = []

        if not sess["system_seeded"]:
            # First call: seed KV cache with long system prompt once
            if self.long_system_prompt:
                ollama_messages.append({"role": "system", "content": self.long_system_prompt})
            # Add memory summary if any (likely empty on first call)
            if sess["memory_summary"]:
                ollama_messages.append({"role": "system", "content": f"Session memory:\n{sess['memory_summary']}"})
            # Add current turn messages
            ollama_messages.extend(current_turn_msgs)

        else:
            # Subsequent calls: tiny system anchor to avoid reprocessing long prompt
            anchor = (
                "You are Mr. Delight, output STRICT JSON only with top-level {\"actions\":[...]}. "
                "Follow the stage rules and update_status for lesson_stage and current_lesson_progress."
            )
            ollama_messages.append({"role": "system", "content": anchor})
            # Include memory summary if exists
            if sess["memory_summary"]:
                ollama_messages.append({"role": "system", "content": f"Session memory:\n{sess['memory_summary']}"})
            # Include short rolling history for recency
            ollama_messages.extend(sess["short_messages"][-2*self.max_window_turns:])
            # Finally, the current turn
            ollama_messages.extend(current_turn_msgs)

            # Attach images to the last user message if any were found
            if images_b64:
                for i in range(len(ollama_messages)-1, -1, -1):
                    if ollama_messages[i]["role"] == "user":
                        ollama_messages[i]["images"] = images_b64  # Ollama expects a list of base64 strings
                        break

        payload = {
            "model": self.model,
            "messages": ollama_messages,
            "stream": False,
            "options": { "temperature": temperature }
        }
        if sess["context"]:
            payload["context"] = sess["context"]

        resp = self._post_chat(payload)

        # Update context
        new_context = resp.get("context")
        if new_context:
            sess["context"] = new_context

        # Append the just-sent user message(s) and the assistant reply into short_messages
        # Current turn user messages are in current_turn_msgs with role 'user' probably twice (summary+instruction + user_action payload)
        # We only append the final user instruction content (last user item) and assistant content
        # But simplest and robust: append all current_turn_msgs and the assistant reply.
        sess["short_messages"].extend(current_turn_msgs)

        ai_text = (resp.get("message", {}) or {}).get("content", "") or ""
        if ai_text:
            sess["short_messages"].append({"role": "assistant", "content": ai_text})

        # Clamp short_messages window
        sess["short_messages"] = sess["short_messages"][-2*self.max_window_turns:]

        # After some growth, summarize memory to compress history
        if len(sess["short_messages"]) > 2 * self.summarize_after_turns:
            self._maybe_summarize(sess)

        # Mark system seeded after first response
        if not sess["system_seeded"]:
            sess["system_seeded"] = True

        # Persist session if enabled
        if self.persist_dir and session_id:
            self._save_session(session_id)

        return ai_text

    def summarize(self, prompt: str, temperature: float = 0.2) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a concise summarizer."},
                {"role": "user", "content": prompt}
            ],
            "stream": False,
            "options": { "temperature": temperature }
        }
        resp = self._post_chat(payload)
        return (resp.get("message", {}) or {}).get("content", "") or ""