# Qwen-2.5-VL OCR Fine-Tuning for Historic Gujarati Documents

This project demonstrates how to fine-tune the Qwen-2.5-VL-3B vision-language model for Optical Character Recognition (OCR) on historic Gujarati document images. The workflow includes dataset preparation, model fine-tuning, evaluation, and inference.

---

## Project Structure

```
qwen2.5vl-ocr-finetune/
│
├── fine-tune_Qwen-2.5-VL-3B_OCR.ipynb        # Main notebook for all steps
├── requirements.txt                          # Python dependencies
├── README.md                                 # This file
│
├── data/
│   ├── assigned_dataset/                     # Provided dataset. 'binarized' folder has enhanced images
│   ├── combined_dataset/                     # Combined data of provided and roboflow open-source data
│   └── roboflow_dataset/                     # Roboflow data
│
├── scripts/
│   ├── infer_finetuned_qwen_vl_3b.py         # Inference script for fine-tuned model
│   └── infer_pretrained_qwen_vl_3b.py        # Inference script for pretrained model
│   |-------------                            # few other processing scripts
|   |                                        
├── metrics_outputs/
│   ├── fine-tune_CER_WER_output.txt
│   ├── finetuned_layout_eval_output.txt
│   ├── pre-train_CER_WER_output.txt
│   ├── pre-trained_layout_eval_output.txt
│   ├── enhanced_pre-train_CER_WER_output.txt         # Results on GAN Enhanced images with pre-trained model
│   └── enhanced_pre-trained_layout_eval_output.txt    # Results on GAN Enhanced images with pre-trained model
│   ├── enhanced_finetune_CER_WER_output.txt         # Results on GAN Enhanced images with finetuned model
│   └── enhanced_finetune_layout_eval_output.txt    # Results on GAN Enhanced images with finetuned model
|
├── model_weight/
│   └── qwen-2-5-vl-3b-checkpoint-1350
│
├── OCR_inference_data/
│   ├── pretrained/
│   │   ├── pretrained_qwen_inference_results.json
│   │   ├── pretrained_qwen_enhanced_inference_results.json
│   │   └── processed_pretrained_qwen_inference_results.json
│   ├── finetuned/
│   │   ├── finetuned_qwen_inference_results.json
│   │   ├── finetuned_qwen_enhanced_inference_results.json
│   │   └── processed_finetuned_qwen_inference_results.json
│   └── enhanced/
│       ├── enhanced_pretrained_qwen_inference_results.json
│       └── enhanced_finetuned_qwen_inference_results.json
│
└── DE-GAN/                                   # (Optional) Image enhancement GAN
```

---

## Environment Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   # Or, from the notebook:
   !pip install bitsandbytes peft trl jiwer qwen-vl-utils[decord]==0.0.8
   ```

2. **(Optional) Install GPU-accelerated PyTorch:**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```
3. **Download processed Dataset and model weights from Google Drive link**

    [Dataset](https://drive.google.com/file/d/1TMAAu35EzpZoVa7BZZVjJIZgzFWnvmIB/view?usp=sharing)

    [Model weights](https://drive.google.com/file/d/1N-Y3z3PKM7g7YBpli1_183iUb3kBU87M/view?usp=sharing)

---

## Dataset Preparation

- Place your images in the appropriate subfolders under `data/`.
- Annotate the images using pytesseract and prepare a JSONL annotation file (e.g., `anno_file.jsonl`) for training:
  - Update input image folder and output JSON path in `annotate_data_with_tessaract.py` and run:
    ```bash
    python scripts/annotate_data_with_tessaract.py
    ```
  - Update input and output JSON path in `conv_to_qwen_format.py` and run: 
    ```bash
    python scripts/conv_to_qwen_format.py
    ```
  - Output file will have entries in the following format:
    ```json
    {"image": "combined_dataset/images/Swarupsannidhan_0001.png", "text": "Gujarati ground truth text"}
    ```
  - Dataset stats:
      Total Number of images-text pairs: 504 
---

## Data Formatting

1. **Example Python snippet:**
   ```python
   import json

   with open("input.jsonl", "r", encoding="utf-8") as fin, open("output.jsonl", "w", encoding="utf-8") as fout:
       for line in fin:
           item = json.loads(line)
           out = {"image": item["image_path"], "text": item["text"]}
           fout.write(json.dumps(out, ensure_ascii=False) + "\n")
   ```

3. **Split the dataset:**
   ```python
   from datasets import Dataset
   data = [ ... ]  # Load your JSONL as a list of dicts
   dataset = Dataset.from_list(data).train_test_split(test_size=0.1)
   ```

---


## Fine-Tuning Qwen-2.5-VL-3B

- Run `fine-tune_Qwen-2.5-VL-3B_OCR.ipynb` for fine-tuning.

1. **Load and pre-process the dataset:**
   ```python
   def load_jsonl(jsonl_path):
       samples = []
       with open(jsonl_path, 'r', encoding='utf-8') as f:
           for line in f:
               item = json.loads(line)
               samples.append({
                   "image": item["image"],
                   "text": item["text"]
               })
       return samples

   data = load_jsonl("/kaggle/working/OCR_project/combined_dataset/anno_file.jsonl")
   ```
   - Split the data into train and eval set:
   ```python
   from datasets import Dataset
   dataset = Dataset.from_list(data).train_test_split(test_size=0.1)
   ```
   - Pre-process and resize the image input.

2. **Format data for chat-based VLM:**
   - Use a function to wrap each sample as a chat message with system/user/assistant roles.
   ```python
   def format_data(sample, image_folder):
       # Preprocess the image
       file_name = sample["image"]
       image = Image.open(os.path.join(image_folder, sample["image"]))
       sample["image"] = preprocess_image(image)

       return {"filename": file_name, "messages": [
           {
               "role": "system",
               "content": [{"type": "text", "text": system_message}],
           },
           {
               "role": "user",
               "content": [
                   {"type": "image", "image": sample["image"]},
                   {"type": "text", "text": "Extract gujarati text from the document images"},
               ],
           },
           {
               "role": "assistant",
               "content": [{"type": "text", "text": sample["text"]}],
           },
       ]}
   ```

3. **Set model and training parameters:**
   ```python
   MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
   EPOCHS = 5
   BATCH_SIZE = 1
   LEARNING_RATE = 2e-5
   # ...
   ```

4. **Load model and processor:**
   ```python
   from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
   model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype="auto", device_map="auto")
   processor = AutoProcessor.from_pretrained(MODEL_ID)
   processor.tokenizer.padding_side = "right"
   ```

5. **Prepare data collator and trainer:**
   - Use a custom collate function to batch images and texts.
   - Use `peft`s methons like `LoRA` for to prepare a model for training on constrained compute resources.
   - Use `trl.SFTTrainer` for training.

6. **Start training:**
   ```python
   trainer.train()
   ```

7. **Save checkpoints:**
   - Checkpoints are saved automatically at intervals.

---

## Inference and Evaluation (WER/CER)

1. **Run batch inference on the validation/test set using the pre-trained model and fine-tuned model:**
   - Download and place checkpoints in model_weight/
   - Loads the fine-tuned model and processor.
   - Processes images and generates text predictions.
   - Saves results to a JSON file.
   ```bash
   python scripts/infer_pretrained_qwen_vl_3b.py
   python scripts/infer_finetuned_qwen_vl_3b.py
   ```
   **Sample output:**
   ```json
   [
     {
       "image_file": "combined_dataset/images/Swarupsannidhan_0001.png",
       "predicted": "Gujarati OCR output",
       "actual": "Gujarati ground truth"
     },
     ...
   ]
   ```

2. **Post-process output JSONs to convert outputs into readable format:**
   ```bash
   python scripts/post_process_json_pretrain.py
   python scripts/post_process_json_finetune.py
   ```

3. **Calculate Word Error Rate (WER) and Character Error Rate (CER):**
   - Run scripts for evaluation:
     ```bash
     python scripts/calculate_WER_CER.py
     python scripts/calculate_structure_sim.py
     ```
   - Example code:
     ```python
     from jiwer import wer
     import editdistance

     sample_wer = wer(actual, predicted)
     sample_cer = editdistance.eval(actual, predicted) / max(len(actual), 1)
     ```

---

## Optional: GAN-based Image Enhancement

- Go to the DE-GAN directory:
  ```bash
  cd qwen2.5vl-ocr-finetune/DE-GAN
  python enhance.py <enhancement type (binarized|deblurred|unwatermark)> <input image folder> <output image folder>
  ```

---

## References

- [Qwen2.5-VL-3B-Instruct on HuggingFace](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- [jiwer (WER metric)](https://github.com/jitsi/jiwer)
- [editdistance (CER metric)](https://github.com/aflc/editdistance)
- [trl (Training Large Language Models)](https://github.com/huggingface/trl)

---

**For further details, see the code and comments in `fine-tune_Qwen-2.5-VL-3B_OCR.ipynb` and the scripts in the `scripts/` directory.**