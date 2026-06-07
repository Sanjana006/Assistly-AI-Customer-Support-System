import os
import torch
import pandas as pd
from datasets import Dataset
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.utils.quantization_config import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl.trainer.sft_config import SFTConfig
from trl.trainer.sft_trainer import SFTTrainer
import argparse

def format_chatml(row):
    """
    Format a dataset row into a ChatML/conversational message template for training.
    Maps customer message to a structured JSON classification response.
    """
    system_prompt = (
        "You are a customer support classifier for an e-commerce platform. "
        "Analyze the customer message and extract intent, sentiment, frustration score, "
        "urgency, any order IDs or emails, and a brief summary. "
        "Always respond with valid JSON matching the exact schema."
    )
    
    # Map raw columns into expected classification structure
    # Standard Bitext columns: instruction, intent, category, response
    # We will format the output response as a clean JSON matching the Classifier schema
    import json
    
    # Map Bitext intents to our system intents
    intent_mapping = {
        "cancel_order": "cancellation_request",
        "check_refund_status": "refund_request",
        "get_refund": "refund_request",
        "track_order": "order_status",
        "check_payment": "payment_issue",
        "complaint": "product_complaint",
    }
    
    raw_intent = row.get("intent", "general_inquiry")
    system_intent = intent_mapping.get(raw_intent, "general_inquiry")
    
    # Infer basic sentiment/frustration based on key indicators
    message = str(row.get("instruction", ""))
    sentiment = "neutral"
    frustration = 0.2
    urgency = "low"
    
    if any(w in message.lower() for w in ["delay", "late", "where is", "waiting"]):
        urgency = "medium"
        frustration = 0.4
    if any(w in message.lower() for w in ["angry", "bad", "worst", "terrible", "frustrated", "refund"]):
        sentiment = "negative"
        frustration = 0.8
        urgency = "high"
        
    # Mock some email/order extractions if present
    order_id = None
    if "ord" in message.lower():
        # find ORDXXXXX
        import re
        match = re.search(r'ord\d{5}', message.lower())
        if match:
            order_id = match.group(0).upper()
            
    structured_output = {
        "intent": system_intent,
        "sentiment": sentiment,
        "frustration_score": frustration,
        "urgency": urgency,
        "order_id": order_id,
        "customer_email": None,
        "summary": row.get("category", "General customer support query")
    }
    
    formatted_response = json.dumps(structured_output)
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Customer message: {message}"},
        {"role": "assistant", "content": formatted_response}
    ]
    return {"messages": messages}

def main(args):
    print("🚀 Initializing QLoRA Fine-tuning pipeline...")
    
    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(
            f"Dataset not found at {args.dataset_path}. "
            f"Please run 'python multi-agent-support/data/load_dataset.py' first."
        )
        
    print(f"Loading dataset from {args.dataset_path}...")
    df = pd.read_csv(args.dataset_path)
    
    # Sample if dataset is too large
    if len(df) > args.max_samples:
        print(f"Sampling {args.max_samples} rows for faster training...")
        df = df.sample(n=args.max_samples, random_state=42)
        
    print(f"Loading base tokenizer: {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Map dataset using tokenizer's apply_chat_template for conversational formatting
    def apply_template(row):
        msg_data = format_chatml(row)
        formatted_text = tokenizer.apply_chat_template(msg_data["messages"], tokenize=False)
        return {"text": formatted_text}

    print("Formatting dataset into chat templates...")
    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(apply_template, remove_columns=dataset.column_names)
    assert isinstance(dataset, Dataset)
    
    # Split training and validation sets
    dataset_split = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset_split["train"]
    eval_dataset = dataset_split["test"]
    
    print(f"Formatted training samples: {len(train_dataset)}")
    print(f"Formatted validation samples: {len(eval_dataset)}")
    
    # Configure 4-bit Quantization (QLoRA)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,  # BF16 for L4/Ada Lovelace GPUs
    )
    
    print(f"Loading base model: {args.base_model}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    
    # Prepare model for PEFT training
    model = prepare_model_for_kbit_training(model)  # type: ignore[operator]
    
    # Configure LoRA Adapters
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Configure SFT Training Parameters
    training_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        optim="paged_adamw_32bit",
        logging_steps=10,
        learning_rate=args.learning_rate,
        fp16=False,
        bf16=True,   # L4 / Ada Lovelace GPUs have native BF16 — faster & more stable than FP16
        max_grad_norm=0.3,
        num_train_epochs=args.epochs,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        report_to="none",
        dataset_text_field="text",
        max_length=512,
        packing=False,
    )
    
    # Initialize SFTTrainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        tokenizer=tokenizer,  # type: ignore[call-arg]
        args=training_args,
    )
    
    print("🔥 Starting QLoRA fine-tuning training loop...")
    trainer.train()
    
    print(f"Saving fine-tuned adapter weights to {args.output_dir}...")
    trainer.save_model(args.output_dir)
    print("✅ Training complete and model adapters successfully saved!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QLoRA Fine-tuning script for Customer Support Classifier")
    parser.add_argument("--dataset_path", type=str, default="data/raw/support_dataset.csv", help="Path to raw support dataset CSV")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="Hugging Face base model identifier")
    parser.add_argument("--output_dir", type=str, default="./fine_tuned_classifier", help="Directory to save fine-tuned adapters")
    parser.add_argument("--max_samples", type=int, default=5000, help="Maximum samples to train on")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per GPU device")
    parser.add_argument("--gradient_accumulation", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank parameter")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha parameter")
    
    args = parser.parse_args()
    main(args)
