import pytesseract
from PIL import Image
import os
import json

# Set Tesseract language
LANG = 'guj'  # Gujarati language code

# Paths
image_folder = '/home/dhruvi/Simform/learning_project/roboflow_dataset/binarized'
output_file = '/home/dhruvi/Simform/learning_project/roboflow_dataset/binarized_output.jsonl'


# Process images
data_entries = []

for filename in os.listdir(image_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(image_folder, filename)
        print(f"Processing {img_path}...")

        # Load and OCR
        image = Image.open(img_path)
        text = pytesseract.image_to_string(image, lang=LANG)
        entry = {
            "image": img_path,
            "text": text.strip(),
            "language": "Gujarati",
            "ocr_source": "tesseract",
            "layout": "single-column",
            "metadata": {
                "filename": filename
            }
        }

        data_entries.append(entry)

# Save as JSONL
with open(output_file, 'w', encoding='utf-8') as f:
    for entry in data_entries:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"\n✅ Dataset saved to: {output_file}")
