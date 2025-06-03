
import json

# Load raw JSON
input_path = "/home/dhruvi/Simform/learning_project/qwen2.5vl-ocr-finetune/OCR_project_data/finetuned_qwen_inference_results.json"
output_path = "/home/dhruvi/Simform/learning_project/qwen2.5vl-ocr-finetune/OCR_project_data/processed_finetuned_qwen_inference_results.json"

with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

def extract_after_assistant_block(text):
    split_token = "assistant\n"
    if split_token in text:
        return text.split(split_token, 1)[-1].strip()
    return text.strip()

results = []
for item in data:
    results.append({
        "image_file": item["image_file"],
        "predicted": extract_after_assistant_block(item["predicted"]),
        "actual": item["actual"].strip()
    })

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"Extracted raw text after 'assistant' saved to: {output_path}")
