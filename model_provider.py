import os
import json
import logging
from typing import List, Dict, Any, Optional
import requests
from io import BytesIO
from PIL import Image
import base64


logger = logging.getLogger(__name__)

# --- Library Imports ---
try:
    import google.generativeai as genai
    from google.generativeai import types as gemini_types
except ImportError:
    genai = None
    gemini_types = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from openai_harmony import (
        Conversation,
        HarmonyEncodingName,
        Message,
        Role,
        SystemContent,
        DeveloperContent, # Import DeveloperContent
        load_harmony_encoding,
        ReasoningEffort
    )
except ImportError:
    Conversation = None
    logger.error("The 'openai-harmony' library is not installed. Please run 'pip install openai-harmony'")


# --- Base Class Definition ---
class ModelProvider:
    def generate_content(self, messages: List[Dict[str, Any]], **kwargs) -> str:
        raise NotImplementedError
    def summarize(self, prompt: str, temperature: float = 0.2) -> str:
        raise NotImplementedError
    def name(self) -> str:
        raise NotImplementedError

# --- Provider Implementations (Gemini and Ollama are unchanged) ---

class GeminiProvider(ModelProvider):
    def __init__(self, api_key: str, model: str = "gemini-1.5-flash-latest"):
        if genai is None: raise RuntimeError("google-generativeai not installed.")
        if not api_key: raise ValueError("Gemini API key is required.")
        genai.configure(api_key=api_key)
        self.model_name = model
        self.model = genai.GenerativeModel(self.model_name)
    def generate_content(self, messages: List[Dict[str, Any]], temperature: float = 0.7, session_id: Optional[str] = None) -> str:
        gemini_messages = []
        for msg in messages:
            role = "model" if msg["role"] == "assistant" else msg["role"]
            parts = [msg["content"]]
            if "image" in msg and msg.get("image"):
                try:
                    img_bytes = base64.b64decode(msg["image"]["b64"])
                    parts.append(Image.open(BytesIO(img_bytes)))
                except Exception as e: logger.error(f"GeminiProvider failed to process image: {e}")
            gemini_messages.append({"role": role, "parts": parts})
        response = self.model.generate_content(
            gemini_messages,
            safety_settings={'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE','HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE','HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE','HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE'},
            generation_config=gemini_types.GenerationConfig(temperature=temperature, candidate_count=1),
        )
        return response.text or ""
    def summarize(self, prompt: str, temperature: float = 0.2) -> str: return self.generate_content([{"role": "user", "content": prompt}], temperature=temperature)
    def name(self) -> str: return f"gemini:{self.model_name}"

class OllamaProvider(ModelProvider):
    def __init__(self, endpoint: str = "http://localhost:11434", model: str = "gemma2:9b"):
        self.endpoint = endpoint.rstrip("/")
        self.chat_url = f"{self.endpoint}/api/chat"
        self.model = model
    def name(self) -> str: return f"ollama:{self.model}@{self.endpoint}"
    def _post_chat(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        r = requests.post(self.chat_url, json=payload, timeout=120)
        r.raise_for_status()
        return r.json()
    def generate_content(self, messages: List[Dict[str, Any]], temperature: float = 0.7, session_id: Optional[str] = None) -> str:
        ollama_messages = []
        for msg in messages:
            ollama_msg = msg.copy()
            if "image" in ollama_msg and ollama_msg.get("image"):
                img_data = ollama_msg.pop("image")
                ollama_msg["images"] = [img_data["b64"]]
            ollama_messages.append(ollama_msg)
        payload = {"model": self.model, "messages": ollama_messages, "stream": False, "options": {"temperature": temperature}}
        resp = self._post_chat(payload)
        return (resp.get("message", {}) or {}).get("content", "") or ""
    def summarize(self, prompt: str, temperature: float = 0.2) -> str:
        messages = [{"role":"system", "content":"You are a concise summarizer."}, {"role":"user", "content":prompt}]
        return self.generate_content(messages, temperature=temperature)

# --- FINAL WORKING NVIDIA PROVIDER ---
class NvidiaProvider(ModelProvider):
    def __init__(self, api_key: str, model: str, developer_prompt: str, output_contract: str):
        if OpenAI is None or Conversation is None:
            raise RuntimeError("Required libraries not found. Please `pip install openai openai-harmony`.")
        if not api_key:
            raise ValueError("NVIDIA API key is required for NvidiaProvider.")
        
        self.model_name = model
        print(self.model_name)
        # **THE FIX**: Store the prompt components.
        self.developer_prompt = developer_prompt
        self.output_contract = output_contract

        self.encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=api_key
        )

    def generate_content(
        self, 
        messages: List[Dict[str, Any]], 
        temperature: float = 1.0, 
        session_id: Optional[str] = None,
        reasoning_effort: str = "medium"
    ) -> str:
        
        effort_map = {"low": ReasoningEffort.LOW, "medium": ReasoningEffort.MEDIUM, "high": ReasoningEffort.HIGH}
        harmony_reasoning_effort = effort_map.get(reasoning_effort.lower(), ReasoningEffort.HIGH)

        harmony_messages = []
        
        # **THE FIX**: Construct the full, structured prompt every time.
        # 1. Add the SystemContent object for reasoning control.
        system_content_obj = SystemContent.new().with_reasoning_effort(harmony_reasoning_effort)
        harmony_messages.append(Message.from_role_and_content(Role.SYSTEM, system_content_obj))

        # 2. Combine the base prompt and output contract into a single developer instruction.
        full_developer_instructions = f"{self.developer_prompt}\n\n{self.output_contract}"
        dev_content = DeveloperContent.new().with_instructions(full_developer_instructions)
        harmony_messages.append(Message.from_role_and_content(Role.DEVELOPER, dev_content))

        # 3. Add the rest of the conversation history, skipping the generic system message.
        for msg in messages:
            if msg.get("role") == "system":
                continue # We've already constructed our own detailed prompt.

            content = msg.get("content", "").strip()
            if not content: continue

            if msg.get("role") in ["assistant", "model"]:
                harmony_messages.append(Message.from_role_and_content(Role.ASSISTANT, content))
            else: # "user"
                harmony_messages.append(Message.from_role_and_content(Role.USER, content))
        
        convo = Conversation.from_messages(harmony_messages)
        prompt_tokens = self.encoding.render_conversation_for_completion(convo, Role.ASSISTANT)
        prompt_string = self.encoding.decode(prompt_tokens)
        # logger.info(f"prompt_string, {prompt_string}")
        try:
            response = self.client.responses.create(
                model=self.model_name,
                input=[prompt_string],
                temperature=1.0,
                max_output_tokens=8192,
            )
            
            if hasattr(response, 'reasoning_text') and response.reasoning_text:
                logger.info(f"NVIDIA Model Reasoning:\n---\n{response.reasoning_text}\n---")
            # logger.info(f"response, {response}")
            raw_response_text = response.output_text if response else ""

            if not raw_response_text:
                logger.warning("NVIDIA API returned OK but no content in 'output_text' for model '%s'.", self.model_name)
                return ""
            
            return raw_response_text.strip()

        except Exception as e:
            logger.error(f"NVIDIA API call failed: {e}", exc_info=True)
            raise

    def summarize(self, prompt: str, temperature: float = 0.2) -> str:
        messages = [{"role": "system", "content": "You are a concise summarizer."}, {"role": "user", "content": prompt}]
        return self.generate_content(messages, temperature=temperature, reasoning_effort="low")

    def name(self) -> str:
        return f"nvidia:{self.model_name}"