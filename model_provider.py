import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple

# Optional imports guarded
try:
    import google.generativeai as genai
    from google.generativeai import types as gemini_types
except Exception:
    genai = None
    gemini_types = None

import requests

logger = logging.getLogger(__name__)

class ModelProvider:
    def generate_content(self, messages: List[Dict[str, Any]], temperature: float = 0.7) -> str:
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

    def generate_content(self, messages: List[Dict[str, Any]], temperature: float = 0.7) -> str:
        # messages format: [{role: user|model, parts: [str, ...]}, ...]
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
    def __init__(self, endpoint: str = "http://localhost:11434", model: str = "gemma3n:e4b", system_prompt: Optional[str] = None):
        self.endpoint = endpoint.rstrip("/")
        self.chat_url = f"{self.endpoint}/api/chat"
        self.model = model
        self.system_prompt = system_prompt or ""
        self.session_context = None  # optional: store context per process; you can also key by session_id if needed.

    def _post_chat(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        r = requests.post(self.chat_url, json=payload, timeout=120)
        r.raise_for_status()
        # Non-streaming returns a single JSON object with "message.content"
        return r.json()

    def generate_content(self, messages: List[Dict[str, Any]], temperature: float = 0.7) -> str:
        # Convert Gemini-like messages into Ollama chat payload
        # Gemini messages: {role, parts: [str...]}
        # Ollama expects: {model, messages: [{role, content}], options, stream}
        ollama_messages = []
        # Insert system prompt at first if not present
        if self.system_prompt:
            ollama_messages.append({"role": "system", "content": self.system_prompt})

        for m in messages:
            role = m.get("role", "user")
            content_parts = m.get("parts", [])
            content = "\n".join([str(p) for p in content_parts if isinstance(p, str)])
            ollama_messages.append({"role": "user" if role == "user" else "assistant", "content": content})

        payload = {
            "model": self.model,
            "messages": ollama_messages,
            "stream": False,
            "options": {
                "temperature": temperature
            }
        }
        if self.session_context:
            payload["context"] = self.session_context

        resp = self._post_chat(payload)
        self.session_context = resp.get("context", self.session_context)
        content = (resp.get("message", {}) or {}).get("content", "") or ""
        return content

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
        if self.session_context:
            payload["context"] = self.session_context

        resp = self._post_chat(payload)
        self.session_context = resp.get("context", self.session_context)
        return (resp.get("message", {}) or {}).get("content", "") or ""

    def name(self) -> str:
        return f"ollama:{self.model}@{self.endpoint}"