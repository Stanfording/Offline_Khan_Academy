Ollama export and serving

1) Convert and package
- If you trained with PEFT/LoRA, merge LoRA into base weights:
  peft-merge:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype="auto")
    peft = PeftModel.from_pretrained(base, "unsloth_gemma3n_sft")
    peft = peft.merge_and_unload()
    peft.save_pretrained("./merged_gemma3n")

- For GGUF (recommended by Ollama):
  Use llama.cpp’s convert-gguf.py (Gemma-compatible) on the merged HuggingFace model:
  python convert.py --framework hf --outfile gemma3n.gguf ./merged_gemma3n
  Then quantize (e.g., Q4_K_M):
  ./quantize gemma3n.gguf gemma3n.Q4_K_M.gguf Q4_K_M

2) Create Modelfile
  FROM ./gemma3n.Q4_K_M.gguf
  PARAM temperature 0.7
  PARAM top_p 0.95
  TEMPLATE """<|system|>
  {system_prompt}

  <|user|>
  {{ .Prompt }}

  <|assistant|>
  """

3) Build and run
  ollama create gemma3n-delight -f Modelfile
  ollama run gemma3n-delight

4) At inference time, pass distilled_prompt.txt as system_prompt (replace token in TEMPLATE or include as a prompt variable).