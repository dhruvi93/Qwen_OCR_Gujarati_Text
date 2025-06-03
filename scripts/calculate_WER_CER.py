import json
import editdistance
from jiwer import wer, cer
from evaluate import load
cer = load("cer")

# Load the cleaned inference results
with open("/home/dhruvi/Simform/learning_project/qwen2.5vl-ocr-finetune/OCR_inference_data/enhanced/processed_finetuned_qwen_inference_enhanced_results.json", "r", encoding="utf-8") as f:
    results = json.load(f)

total_wer = 0
total_cer = 0
sample_count = 0

for item in results:
    predicted = item.get("predicted", "")
    actual = item.get("actual", "")
    image_file = item.get("image_file", "unknown")

    predicted = predicted.strip() if isinstance(predicted, str) else ""
    actual = actual.strip() if isinstance(actual, str) else ""

    # Skip if both are empty or not strings
    if not predicted and not actual:
        continue

    # Compute Word Error Rate (WER)
    try:
        sample_wer = wer(actual, predicted)
    except Exception as e:
        print(f"WER computation failed for image: {image_file} ({e})")
        sample_wer = 1.0  # worst case

    print(f"sample_wer: {sample_wer:.4f} for image: {image_file}")

    # Compute Character Error Rate (CER)
    try:
        # Using editdistance for CER computation
        # distance = editdistance.eval(actual, predicted)
        # cer = distance / max(len(actual), 1)

        # Using evaluate for CER computation
        cers = cer.compute(predictions=[predicted], references=[actual])
    except Exception as e:
        print(f"CER computation failed for image: {image_file} ({e})")
        cers = 1.0

    print(f"sample_cer: {cers:.4f} for image: {image_file}")

    total_wer += sample_wer
    total_cer += cers
    sample_count += 1

if sample_count == 0:
    print("No valid samples found for evaluation.")
else:
    average_wer = total_wer / sample_count
    average_cer = total_cer / sample_count

    print(f" Evaluated {sample_count} samples")
    print(f"Average Word Error Rate (WER): {average_wer:.4f}")
    print(f"Average Character Error Rate (CER): {average_cer:.4f}")
