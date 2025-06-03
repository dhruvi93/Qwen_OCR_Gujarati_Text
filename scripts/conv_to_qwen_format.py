import json

input_path = "/home/dhruvi/Simform/learning_project/qwen2.5vl-ocr-finetune/data/assigned_dataset/binarized_output.jsonl"
output_path = "/home/dhruvi/Simform/learning_project/qwen2.5vl-ocr-finetune/data/assigned_dataset/binarize_qwen_vl_out.jsonl"

with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
    for line in fin:
        entry = json.loads(line)
        out_entry = {
            "image": f"combined_enhanced_data/images/{entry['metadata']['filename']}",
            "text": entry["text"]
        }
        fout.write(json.dumps(out_entry, ensure_ascii=False) + "\n")