import os
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import json
from datetime import datetime

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR = "./qwen_output"


def load_model_and_tokenizer(model_path=None, device="cpu"):
    if model_path is None:
        model_path = MODEL_NAME

    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float32, device_map=device, trust_remote_code=True
    )
    model.config.use_cache = False

    return model, tokenizer


def generate_response(model, tokenizer, question, device):
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": question},
    ]

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer([text], return_tensors="pt").to(device)

    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "assistant\n" in response:
        response = response.split("assistant\n")[-1]
    elif "assistant" in response:
        response = response.split("assistant")[-1]

    return response.strip()


def calculate_metrics(prediction, ground_truth):
    pred_lower = prediction.lower().strip()
    truth_lower = ground_truth.lower().strip()

    exact_match = 1.0 if pred_lower == truth_lower else 0.0

    pred_words = set(pred_lower.split())
    truth_words = set(truth_lower.split())

    if len(truth_words) == 0:
        precision = 0.0
        recall = 0.0
        f1 = 0.0
    else:
        intersection = pred_words.intersection(truth_words)
        precision = len(intersection) / len(pred_words) if len(pred_words) > 0 else 0.0
        recall = len(intersection) / len(truth_words)
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

    return {
        "exact_match": exact_match,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate_model(model, tokenizer, df, device):
    results = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        question = str(row["input"]).strip()
        ground_truth = str(row["target"]).strip()

        prediction = generate_response(model, tokenizer, question, device)
        metrics = calculate_metrics(prediction, ground_truth)

        results.append(
            {
                "question": question,
                "ground_truth": ground_truth,
                "prediction": prediction,
                "metrics": metrics,
            }
        )

    avg_metrics = {
        "exact_match": sum(r["metrics"]["exact_match"] for r in results) / len(results),
        "precision": sum(r["metrics"]["precision"] for r in results) / len(results),
        "recall": sum(r["metrics"]["recall"] for r in results) / len(results),
        "f1": sum(r["metrics"]["f1"] for r in results) / len(results),
    }

    return results, avg_metrics


def main():
    print("=" * 60)
    print("Model Comparison: Before vs After Fine-tuning")
    print("=" * 60)

    device = "cpu"

    df = pd.read_csv("dataset.csv")
    df = df.dropna()
    df = df.head(10)

    print(f"\nTest dataset size: {len(df)}")

    print("\n" + "=" * 60)
    print("STEP 1: Evaluate BEFORE Fine-tuning")
    print("=" * 60)

    base_model, base_tokenizer = load_model_and_tokenizer(MODEL_NAME, device)

    before_results, before_metrics = evaluate_model(
        base_model, base_tokenizer, df, device
    )

    print(f"\nBefore Fine-tuning Metrics:")
    print(f"  Exact Match: {before_metrics['exact_match']:.4f}")
    print(f"  Precision:  {before_metrics['precision']:.4f}")
    print(f"  Recall:     {before_metrics['recall']:.4f}")
    print(f"  F1 Score:  {before_metrics['f1']:.4f}")

    del base_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print("\n" + "=" * 60)
    print("STEP 2: Evaluate AFTER Fine-tuning")
    print("=" * 60)

    trained_model, trained_tokenizer = load_model_and_tokenizer(OUTPUT_DIR, device)

    after_results, after_metrics = evaluate_model(
        trained_model, trained_tokenizer, df, device
    )

    print(f"\nAfter Fine-tuning Metrics:")
    print(f"  Exact Match: {after_metrics['exact_match']:.4f}")
    print(f"  Precision:  {after_metrics['precision']:.4f}")
    print(f"  Recall:     {after_metrics['recall']:.4f}")
    print(f"  F1 Score:  {after_metrics['f1']:.4f}")

    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)

    print("\nMetric Comparison:")
    print(f"{'Metric':<15} {'Before':<12} {'After':<12} {'Change':<12}")
    print("-" * 51)

    changes = {}
    for metric in ["exact_match", "precision", "recall", "f1"]:
        before_val = before_metrics[metric]
        after_val = after_metrics[metric]
        change = after_val - before_val
        changes[metric] = change

        change_str = f"+{change:.4f}" if change >= 0 else f"{change:.4f}"
        print(f"{metric:<15} {before_val:<12.4f} {after_val:<12.4f} {change_str:<12}")

    comparison_results = {
        "timestamp": datetime.now().isoformat(),
        "model_name": MODEL_NAME,
        "test_size": len(df),
        "before_finetuning": {"metrics": before_metrics, "samples": before_results[:3]},
        "after_finetuning": {"metrics": after_metrics, "samples": after_results[:3]},
        "improvement": changes,
    }

    output_file = os.path.join(OUTPUT_DIR, "comparison_results.json")
    with open(output_file, "w") as f:
        json.dump(comparison_results, f, indent=2, ensure_ascii=False)

    print(f"\nComparison results saved to: {output_file}")

    print("\n" + "=" * 60)
    print("Sample Comparisons")
    print("=" * 60)

    for i in range(min(3, len(df))):
        print(f"\n--- Sample {i + 1} ---")
        print(f"Question:      {before_results[i]['question'][:80]}...")
        print(f"Ground Truth:  {before_results[i]['ground_truth']}")
        print(f"Before:       {before_results[i]['prediction'][:80]}...")
        print(f"After:        {after_results[i]['prediction'][:80]}...")

    print("\n" + "=" * 60)
    print("Comparison Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
