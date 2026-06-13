import sys, os
from loguru import logger
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from config import Config

_cache = {}

def get_local_classifier(adapter_path: str):
    global _cache
    if "_local_model" not in _cache or _cache.get("_local_model_path") != adapter_path:
        logger.info(f"Local Classifier: Loading Qwen2.5-1.5B base + LoRA adapter from {adapter_path}...")

        BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

        tokenizer = AutoTokenizer.from_pretrained(adapter_path)  # type: ignore[misc]
        tokenizer.pad_token = tokenizer.eos_token  # type: ignore[assignment]
        
        chat_template_path = os.path.join(adapter_path, "chat_template.jinja")
        if os.path.exists(chat_template_path) and not getattr(tokenizer, "chat_template", None):
            with open(chat_template_path, "r", encoding="utf-8") as f:
                tokenizer.chat_template = f.read()

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        base_model = AutoModelForCausalLM.from_pretrained(  # type: ignore[misc]
            BASE_MODEL,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        if base_model is None:
            raise RuntimeError(f"AutoModelForCausalLM.from_pretrained returned None for {BASE_MODEL}")
        model = PeftModel.from_pretrained(base_model, adapter_path)
        model.eval()

        _cache["_local_model"]      = model
        _cache["_local_tokenizer"]  = tokenizer
        _cache["_local_model_path"] = adapter_path
        logger.info("Local Classifier: Model loaded and cached.")

    return _cache["_local_model"], _cache["_local_tokenizer"]
