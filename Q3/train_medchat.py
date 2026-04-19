import os
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset, load_dataset
from trl import SFTTrainer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR = "./medchat_output"

TRAIN_SIZE = 100
MAX_LENGTH = 256
LORA_R = 4


def format_conversation(example):
    patient_msg = str(example.get("patient_message", ""))[:MAX_LENGTH]
    doctor_response = str(example.get("doctor_response", ""))[:MAX_LENGTH]
    return {"text": f"Patient: {patient_msg}\n\nDoctor: {doctor_response}<|endoftext|>"}


def main():
    print("=" * 60)
    print("Medical Chatbot Fine-tuning Pipeline")
    print("=" * 60)

    print("\n[1/6] Loading dataset from Hugging Face...")
    ds = load_dataset("OpenMed/MedDialog")
    print(f"  Total samples: {len(ds['train'])}")

    print(f"\n[2/6] Sampling {TRAIN_SIZE} rows...")
    df = ds["train"].to_pandas()
    df = df.sample(n=min(TRAIN_SIZE, len(df)), random_state=42)
    print(f"  Using: {len(df)} samples")

    print("\n[3/6] Formatting data...")
    formatted = df.apply(format_conversation, axis=1).tolist()
    dataset = Dataset.from_list(formatted)
    print(f"  Formatted: {len(dataset)} samples")

    print("\n[4/6] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32, device_map="cpu", trust_remote_code=True
    )
    model.config.use_cache = False
    print(f"  Model: {model.num_parameters():,} params")

    print("\n[5/6] Configuring LoRA...")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_R * 2,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("\n[6/6] Training...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=3e-4,
        fp16=False,
        logging_steps=20,
        save_steps=50,
        save_total_limit=1,
        warmup_steps=5,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        formatting_func=lambda x: x["text"],
    )

    print("-" * 60)
    trainer.train()

    print("\nSaving model...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model saved to {OUTPUT_DIR}")

    print("\n" + "=" * 60)
    print("Testing inference...")
    print("=" * 60)

    test_question = "I have a severe headache for 3 days. What should I do?"
    print(f"Question: {test_question}")

    messages = [
        {"role": "system", "content": "You are a medical assistant."},
        {"role": "user", "content": test_question},
    ]

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    device = next(model.parameters()).device
    inputs = tokenizer([text], return_tensors="pt").to(device)

    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=256, do_sample=True, temperature=0.7
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(f"Answer: {response}")

    print("=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
