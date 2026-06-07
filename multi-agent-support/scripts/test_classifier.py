#!/usr/bin/env python3
import os
import sys
import time
import json
import re
import torch
import argparse
from typing import Any
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.utils.quantization_config import BitsAndBytesConfig
from peft import PeftModel
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Ensure parent directory is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

console = Console()

def detect_device():
    """Detects the best available hardware accelerator."""
    if torch.cuda.is_available():
        return "cuda", True
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # Apple Silicon Mac GPU
        return "mps", False
    else:
        return "cpu", False

def load_local_classifier(adapter_path, base_model_name="Qwen/Qwen2.5-1.5B-Instruct"):
    """Loads the base model and Peft adapters, choosing the best device configuration."""
    device, use_4bit = detect_device()
    
    console.print(f"\n[bold blue]Hardware Detection:[/bold blue]")
    console.print(f"  - Device: [bold green]{device.upper()}[/bold green]")
    console.print(f"  - 4-bit Quantization (bitsandbytes): [bold green]{'Enabled' if use_4bit else 'Disabled (not supported/needed on this device)'}[/bold green]")
    
    console.print(f"\n[bold yellow]Loading model and tokenizer...[/bold yellow]")
    start_time = time.time()
    
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load chat template from jinja file if present and not loaded automatically
    chat_template_path = os.path.join(adapter_path, "chat_template.jinja")
    if os.path.exists(chat_template_path) and not getattr(tokenizer, "chat_template", None):
        with open(chat_template_path, "r", encoding="utf-8") as f:
            tokenizer.chat_template = f.read()
    
    model_kwargs: dict[str, Any] = {}
    
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        model_kwargs["quantization_config"] = bnb_config  # type: ignore[assignment]
        model_kwargs["torch_dtype"] = torch.float16  # type: ignore[assignment]
        model_kwargs["device_map"] = "auto"
    else:
        if device == "mps":
            model_kwargs["torch_dtype"] = torch.float16  # type: ignore[assignment]
        else:
            model_kwargs["torch_dtype"] = torch.float32  # type: ignore[assignment]  # CPU is safer/more stable in FP32
        model_kwargs["device_map"] = None
            
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        **model_kwargs
    )
    
    console.print(f"Loading LoRA adapters from [bold green]{adapter_path}[/bold green]...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    if not use_4bit:
        console.print(f"Moving model to [bold green]{device.upper()}[/bold green]...")
        model = model.to(device)
        
    model.eval()
    
    elapsed = time.time() - start_time
    console.print(f"[bold green]✅ Model loaded successfully in {elapsed:.2f}s![/bold green]\n")
    return model, tokenizer, device

def run_inference(model, tokenizer, device, message):
    """Formats the message, runs inference, and parses the JSON response."""
    system_prompt = (
        "You are a customer support classifier for an e-commerce platform. "
        "Analyze the customer message and extract intent, sentiment, frustration score, "
        "urgency, any order IDs or emails, and a brief summary. "
        "Always respond with valid JSON matching the exact schema."
    )
    
    chat_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Customer message: {message}"},
    ]
    
    prompt_text = tokenizer.apply_chat_template(
        chat_messages, tokenize=False, add_generation_prompt=True
    )
    
    inputs = tokenizer(prompt_text, return_tensors="pt")
    # Move inputs to correct device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    start_time = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    generation_time = (time.time() - start_time) * 1000
    
    # Extract only the newly generated tokens
    prompt_length = inputs["input_ids"].shape[-1]
    generated_tokens = output_ids[0][prompt_length:]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    
    # Parse JSON
    json_match = re.search(r"\{.*\}", generated_text, re.DOTALL)
    parsed_json = None
    parse_error = None
    
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            if isinstance(parsed, dict):
                parsed_json = parsed
                # Normalize frustration score to 0.0 - 1.0 range
                if "frustration_score" in parsed_json:
                    parsed_json["frustration_score"] = max(0.0, min(1.0, float(parsed_json.get("frustration_score", 0.0))))
            else:
                parse_error = "Decoded JSON is not a dictionary."
        except Exception as e:
            parse_error = str(e)
    else:
        parse_error = "Could not locate JSON object block `{...}` in output."
        
    return {
        "raw_output": generated_text,
        "parsed_json": parsed_json,
        "parse_error": parse_error,
        "latency_ms": generation_time
    }

def print_result(message, result):
    """Outputs the test result using rich styling."""
    console.print(Panel(f"[bold]Input Message:[/bold] {message}", border_style="cyan"))
    
    parsed_json = result.get("parsed_json")
    if isinstance(parsed_json, dict):
        table = Table(title="Extracted Classification Fields", show_header=True, header_style="bold magenta")
        table.add_column("Field", style="cyan")
        table.add_column("Value", style="green")
        
        for k, v in parsed_json.items():
            table.add_row(k, str(v))
            
        console.print(table)
    else:
        console.print(f"[bold red]❌ JSON Parsing Failed:[/bold red] {result['parse_error']}")
        console.print(Panel(result["raw_output"], title="[red]Raw Generated Output[/red]", border_style="red"))
        
    console.print(f"⏱️ [dim]Generation Time: {result['latency_ms']:.0f}ms[/dim]\n")
    console.print("-" * 50)

def main():
    parser = argparse.ArgumentParser(description="Test local fine-tuned classifier adapter model.")
    parser.add_argument(
        "--adapter_path", 
        type=str, 
        default="./fine_tuned_classifier", 
        help="Path to the directory containing fine-tuned adapters (default: ./fine_tuned_classifier)"
    )
    parser.add_argument(
        "--base_model", 
        type=str, 
        default="Qwen/Qwen2.5-1.5B-Instruct", 
        help="Base model used for training (default: Qwen/Qwen2.5-1.5B-Instruct)"
    )
    parser.add_argument(
        "--message", 
        type=str, 
        help="Custom customer message to classify"
    )
    parser.add_argument(
        "--interactive", 
        action="store_true", 
        help="Run in interactive shell mode"
    )
    
    args = parser.parse_args()
    
    # Search for adapter path in common candidate locations
    candidates = [
        args.adapter_path,
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fine_tuned_classifier"),
        os.path.join("multi-agent-support", "fine_tuned_classifier"),
    ]
    
    adapter_path = None
    for candidate in candidates:
        if os.path.exists(candidate) and len(os.listdir(candidate)) > 0:
            adapter_path = candidate
            break
            
    if not adapter_path:
        console.print(f"[bold red]Error:[/bold red] No model weights/adapters found at any of the candidate locations:")
        for c in candidates:
            console.print(f"  - {os.path.abspath(c)}")
        console.print("\nPlease make sure you have run the fine-tuning script or have valid adapters placed there.")
        sys.exit(1)
        
    # Load model
    model, tokenizer, device = load_local_classifier(adapter_path, args.base_model)
    
    # If a single message is specified
    if args.message:
        result = run_inference(model, tokenizer, device, args.message)
        print_result(args.message, result)
        return

    # Interactive mode
    if args.interactive:
        console.print("[bold green]Interactive Mode Active.[/bold green] Enter your messages to classify (type 'exit' or 'quit' to stop).\n")
        while True:
            try:
                msg = input("Customer message > ")
                if msg.strip().lower() in ["exit", "quit"]:
                    break
                if not msg.strip():
                    continue
                result = run_inference(model, tokenizer, device, msg)
                print_result(msg, result)
            except KeyboardInterrupt:
                break
        return

    # Default batch of test cases (covering various intents and edge cases)
    test_cases = [
        "Hi, where is my order ORD00042? It was supposed to arrive 3 days ago.",
        "THIS IS RIDICULOUS!! I ordered ORD00015 TWO WEEKS AGO and it still hasnt arrived! I WANT A FULL REFUND NOW!!",
        "What is your return policy for electronics?",
        "I want to talk to a real person please.",
        "Can I update my email address on my account? It is sanjana@example.com."
    ]
    
    console.print("[bold cyan]Running default test suite scenarios...[/bold cyan]\n")
    for msg in test_cases:
        result = run_inference(model, tokenizer, device, msg)
        print_result(msg, result)

if __name__ == "__main__":
    main()
