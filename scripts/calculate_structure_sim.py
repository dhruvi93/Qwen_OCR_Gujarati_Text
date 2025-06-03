import json
import difflib

def normalize(text):
    return text.strip().replace("\r\n", "\n").replace("\r", "\n")

def get_lines(text):
    return [line.strip() for line in normalize(text).split("\n") if line.strip()]

def linewise_accuracy(actual, predicted):
    actual_lines = get_lines(actual)
    predicted_lines = get_lines(predicted)

    match_count = 0
    total = max(len(actual_lines), 1)

    for act_line, pred_line in zip(actual_lines, predicted_lines):
        if act_line == pred_line:
            match_count += 1

    return match_count / total

def structural_token_accuracy(actual, predicted):
    actual_tokens = normalize(actual).count("\n")
    predicted_tokens = normalize(predicted).count("\n")
    if actual_tokens == 0:
        return 1.0 if predicted_tokens == 0 else 0.0
    return 1.0 - abs(predicted_tokens - actual_tokens) / actual_tokens

def layout_aware_accuracy(actual, predicted, weight_line=0.7, weight_struct=0.3):
    line_acc = linewise_accuracy(actual, predicted)
    struct_acc = structural_token_accuracy(actual, predicted)
    return weight_line * line_acc + weight_struct * struct_acc

# Load cleaned predictions
with open("/home/dhruvi/Simform/learning_project/qwen2.5vl-ocr-finetune/OCR_inference_data/processed_pretrained_qwen_enhanced_inference_results.json", "r", encoding="utf-8") as f:
    results = json.load(f)

layout_scores = []

for item in results:
    pred = item.get("predicted", "")
    act = item.get("actual", "")
    image_file = item.get("image_file", "unknown")
    score = layout_aware_accuracy(act, pred)
    layout_scores.append(score)
    
    print(f"layout score : {score:.4f} for image: {image_file}")
    
if layout_scores:
    avg_score = sum(layout_scores) / len(layout_scores)
    print(f" Evaluated {len(layout_scores)} samples")
    print(f"Average Layout-Aware Sequence Accuracy: {avg_score:.4f}")
else:
    print("No valid samples found for evaluation.")
