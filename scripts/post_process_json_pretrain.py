import json
import re

# Load the raw inference results
with open('/home/dhruvi/Simform/learning_project/qwen2.5vl-ocr-finetune/OCR_project_data/finetuned_qwen_inference_results.json', 'r', encoding='utf-8') as f:
    raw_data = json.load(f)

cleaned_results = []

# Helper: extract only the Gujarati text after 'assistant\n'
def extract_gujarati(text):
    if 'assistant' in text:
        return text.split('assistant', 1)[-1].strip()
    return text.strip()

# Clean and prepare new results
for item in raw_data:
    # print(item["actual"].strip())
    cleaned_results.append({
        "image_file": item["image_file"],
        "predicted": extract_gujarati(item["predicted"]),
        "actual": item["actual"].strip()
    })

# Save to a new cleaned JSON file
with open('/home/dhruvi/Simform/learning_project/qwen2.5vl-ocr-finetune/OCR_project_data/processed_finetuned_qwen_inference_results.json', 'w', encoding='utf-8') as f:
    json.dump(cleaned_results, f, ensure_ascii=False, indent=2)

print("processed predictions saved")