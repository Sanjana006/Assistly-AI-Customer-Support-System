from langchain_groq import ChatGroq          # ← was: from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, SecretStr
from typing import Optional, Literal
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

def safe_invoke(llm, messages, retries: int = 3, backoff: int = 2):
    attempt = 0
    while True:
        try:
            return llm.invoke(messages)
        except Exception as e:
            is_rate_limit = "rate limit" in str(e).lower() or "429" in str(e) or "ratelimit" in e.__class__.__name__.lower()
            if is_rate_limit and attempt < retries:
                attempt += 1
                from loguru import logger
                logger.warning(f"Groq rate limit hit in classifier, retrying in {backoff}s... Attempt {attempt}/{retries}")
                import time
                time.sleep(backoff)
                backoff *= 2
                continue
            else:
                raise

class ClassificationResult(BaseModel):
    intent: Literal[
        "order_status", "refund_request", "shipping_inquiry",
        "product_complaint", "account_issue", "payment_issue",
        "cancellation_request", "general_inquiry", "human_request"
    ] = Field(description="The primary intent of the customer message")
    sentiment: Literal["positive", "neutral", "negative", "angry"]
    frustration_score: float = Field(ge=0.0, le=1.0)
    urgency: Literal["low", "medium", "high"]
    order_id: Optional[str] = Field(default=None)
    customer_email: Optional[str] = Field(default=None)
    summary: str

def classify_message(state: dict) -> dict:
    existing_order_id = state.get("classification", {}).get("order_id")
    result_dict = None  # Will be set by local PEFT model if available, else Cloud API

    # ── Local PEFT Classifier (only if enabled AND weights exist) ───────────
    if Config.USE_LOCAL_CLASSIFIER:
        from loguru import logger
        adapter_path = Config.LOCAL_MODEL_PATH
        if os.path.exists(adapter_path) and len(os.listdir(adapter_path)) > 0:
            try:
                import json, re, torch
                from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
                from peft import PeftModel

                # ── Cache model across requests (load once, reuse) ──────────
                # We stash the loaded model on this module so Streamlit reruns
                # don't reload 1.5B weights on every message.
                _cache = sys.modules[__name__].__dict__
                if "_local_model" not in _cache or _cache.get("_local_model_path") != adapter_path:
                    logger.info(f"Local Classifier: Loading Qwen2.5-1.5B base + LoRA adapter from {adapter_path}...")

                    BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

                    tokenizer = AutoTokenizer.from_pretrained(adapter_path)  # type: ignore[misc]
                    tokenizer.pad_token = tokenizer.eos_token  # type: ignore[assignment]
                    
                    # Load chat template from jinja file if present and not loaded automatically
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

                model     = _cache["_local_model"]
                tokenizer = _cache["_local_tokenizer"]

                # ── Build the same prompt used during training ───────────────
                system_prompt = (
                    "You are a customer support classifier for an e-commerce platform. "
                    "Analyze the customer message and extract intent, sentiment, frustration score, "
                    "urgency, any order IDs or emails, and a brief summary. "
                    "Always respond with valid JSON matching the exact schema."
                )
                chat_messages = [
                    {"role": "system",    "content": system_prompt},
                    {"role": "user",      "content": f"Customer message: {state['user_message']}"},
                ]
                prompt_text = tokenizer.apply_chat_template(
                    chat_messages, tokenize=False, add_generation_prompt=True
                )
                inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

                logger.info("Local Classifier: Running inference with LoRA adapter...")
                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=200,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                generated = tokenizer.decode(
                    output_ids[0][inputs["input_ids"].shape[-1]:],
                    skip_special_tokens=True
                ).strip()

                # ── Parse JSON out of the generated text ─────────────────────
                json_match = re.search(r"\{.*\}", generated, re.DOTALL)
                if json_match:
                    parsed = json.loads(json_match.group())
                    parsed["frustration_score"] = max(0.0, min(1.0, float(parsed.get("frustration_score", 0.2))))
                    result = ClassificationResult(**parsed)
                    result_dict = result.model_dump()
                    logger.info(f"Local Classifier: Intent={result_dict['intent']}, Sentiment={result_dict['sentiment']}")
                else:
                    logger.warning(f"Local Classifier: Could not parse JSON from output: {generated!r}. Falling back.")

            except Exception as e:
                from loguru import logger as _log
                _log.warning(f"Local Classifier: Failed ({e}). Falling back to Cloud API.")
        else:
            from loguru import logger
            logger.warning(
                f"Local Classifier: No weights at {adapter_path}. "
                "Run scripts/finetune.py first. Falling back to Cloud API."
            )

    # Cloud API Classifier — uses JSON mode to avoid Groq tool_use_failed bug
    # llama-3.1-8b-instant wraps structured output in <function=...> tags when
    # using with_structured_output, so we inject the schema into the prompt and
    # parse the raw JSON response manually instead.
    if not result_dict:
        import json
        api_key = SecretStr(Config.GROQ_API_KEY) if Config.GROQ_API_KEY else None
        llm = ChatGroq(
            model=Config.get_model(),
            api_key=api_key,
            max_tokens=500,
            model_kwargs={"response_format": {"type": "json_object"}}
        )

        schema_str = """{
  "intent": one of ["order_status","refund_request","shipping_inquiry","product_complaint","account_issue","payment_issue","cancellation_request","general_inquiry","human_request"],
  "sentiment": one of ["positive","neutral","negative","angry"],
  "frustration_score": float between 0.0 and 1.0,
  "urgency": one of ["low","medium","high"],
  "order_id": string of the format ORD followed by digits (e.g. ORD00042) if explicitly mentioned in the message, else null,
  "customer_email": string if mentioned, else null,
  "summary": short string summarising the request
}"""

        system_prompt = f"""You are a customer support classifier for an e-commerce platform.
Analyse the customer message and respond ONLY with a valid JSON object matching this exact schema:
{schema_str}

Rules:
- intent must be one of the listed values only
- frustration_score must be a number between 0.0 and 1.0
- order_id must match pattern ORD followed by digits (e.g. ORD00042) or null
- Return ONLY the JSON object, no extra text, no markdown, no wrapping tags"""

        try:
            response = safe_invoke(llm, [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Customer message: {state['user_message']}")
            ])

            raw = str(response.content).strip()
            # Strip any accidental markdown code fences
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()

            parsed = json.loads(raw)
            # Clamp frustration_score to valid range and validate via Pydantic
            parsed["frustration_score"] = max(0.0, min(1.0, float(parsed.get("frustration_score", 0.0))))
            result = ClassificationResult(**parsed)
            result_dict = result.model_dump()
        except Exception as e:
            from loguru import logger
            logger.error(f"Classifier Cloud API failed: {e}. Using fallback classification.")
            result_dict = {
                "intent": "general_inquiry",
                "sentiment": "neutral",
                "frustration_score": 0.0,
                "urgency": "low",
                "order_id": existing_order_id,
                "customer_email": None,
                "summary": "Fallback classification due to API error"
            }

    # Ensure order ID is valid and not hallucinated or incorrectly defaulted
    import re
    user_msg = state["user_message"]
    
    # 1. Direct regex match (guaranteed extraction)
    regex_match = re.search(r"\b(ORD\d+)\b", user_msg, re.IGNORECASE)
    if regex_match:
        raw_id = regex_match.group(1).upper()
        digits_part = re.search(r"\d+", raw_id)
        if digits_part:
            result_dict["order_id"] = f"ORD{digits_part.group().zfill(5)}"
        else:
            result_dict["order_id"] = raw_id
    else:
        # 2. If the LLM extracted an order ID, verify it is actually referred to in the message
        llm_order_id = result_dict.get("order_id")
        if isinstance(llm_order_id, str):
            # Extract digits from LLM order ID (e.g. 42 from ORD00042)
            digits_match = re.search(r"\d+", llm_order_id)
            if digits_match:
                digits = digits_match.group().lstrip("0")
                if not digits:
                    digits = "0"
                # Check if digits or order ID is in the message
                if digits in user_msg or llm_order_id.lower() in user_msg.lower():
                    result_dict["order_id"] = f"ORD{digits_match.group().zfill(5)}"
                else:
                    result_dict["order_id"] = None
            else:
                result_dict["order_id"] = None
        else:
            result_dict["order_id"] = None

    # Preserve pinned active order ID if the model did not detect a new one
    if not result_dict.get("order_id") and existing_order_id:
        result_dict["order_id"] = existing_order_id

    state["classification"]    = result_dict
    state["intent"]            = result_dict["intent"]
    state["sentiment"]         = result_dict["sentiment"]
    state["frustration_score"] = result_dict["frustration_score"]
    return state