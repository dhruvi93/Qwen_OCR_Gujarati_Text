import io
from PIL import Image
from io import BytesIO
import base64
import os
os.environ["WANDB_DISABLED"] = "true"

# from datasets import load_dataset
import torch
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer

import warnings
warnings.filterwarnings("ignore")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# system_message = """You are a highly advanced Vision Language Model (VLM), specialized in extracting the text from historical documents. Extract the gujarati text from scanned historical document"""

system_message = """
You are a highly advanced Vision Language Model (VLM), specialized in extracting Gujarati text from scanned historical documents. 

Extract and return only the Gujarati text content, without any additional commentary, explanation, formatting, or headings. 

Your response should contain only the raw extracted Gujarati text.
"""


## pre-process data
def resize_image(image, max_size=(512, 512)):
    """
    Resize an image to a maximum size while maintaining aspect ratio.
    """
    image.thumbnail(max_size, Image.Resampling.LANCZOS)  # Updated here
    return image

def compress_image(image, quality=95):
    """
    Compress an image by reducing its quality.
    """
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    compressed_image = Image.open(buffer)
    return compressed_image

def preprocess_image(image, max_size=(812, 812), quality=85):
    """
    Preprocess an image by resizing and compressing it.
    """
    resized_image = resize_image(image, max_size)
    compressed_image = compress_image(resized_image, quality)
    return compressed_image


def format_data(sample,image_folder):
  # Preprocess the image
  file_name = sample["image"]
  image = Image.open(os.path.join(image_folder,sample["image"]))
  sample["image"] = preprocess_image(image)

  return {"filename" :file_name, "messages":[
      {
          "role": "system",
          "content": [{"type": "text", "text": system_message}],
      },
      {
          "role": "user",
          "content": [
              {
                  "type": "image",
                  "image": sample["image"],
              },
              {
                  "type": "text",
                  "text": "Extract gujarati text from the document images",
              },
              
          ],
      },
      {
          "role": "assistant",
          "content": [{"type": "text", "text": sample["text"]}],
      },
   ]}
  
  
## load and prepare dataset
import os
import json
from PIL import Image
from datasets import load_dataset, Dataset
from transformers import AutoProcessor, AutoModelForVision2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments

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

data = load_jsonl("data/combined_dataset/anno_file.jsonl")
dataset = Dataset.from_list(data).train_test_split(test_size=0.1)

image_folder = "data"
train_dataset, eval_dataset = dataset['train'], dataset['test']

train_dataset = [format_data(sample,image_folder) for sample in train_dataset]
eval_dataset = [format_data(sample,image_folder) for sample in eval_dataset]


# sample_data = eval_dataset[0]
# sample_question = sample_data["messages"][1]["content"][1]["text"]
# sample_image = sample_data["messages"][1]["content"][0]["image"]
# sample_answer = sample_data["messages"][2]["content"][0]["text"]


## define model params
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
EPOCHS = 5
BATCH_SIZE = 1
GRADIENT_CHECKPOINTING = True,
USE_REENTRANT = False,
OPTIM = "paged_adamw_32bit"
LEARNING_RATE = 2e-5
LOGGING_STEPS = 50
EVAL_STEPS = 50
SAVE_STEPS = 50
EVAL_STRATEGY = "steps"
SAVE_STRATEGY = "steps"
METRIC_FOR_BEST_MODEL="eval_loss"
LOAD_BEST_MODEL_AT_END=True
MAX_GRAD_NORM = 1
WARMUP_STEPS = 0
DATASET_KWARGS={"skip_prepare_dataset": True}
REMOVE_UNUSED_COLUMNS = False
MAX_SEQ_LEN=3048
NUM_STEPS = (283 // BATCH_SIZE) * EPOCHS
print(f"NUM_STEPS: {NUM_STEPS}")


## load model
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

# default: Load the model on the available device(s)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID, torch_dtype="auto", device_map="auto"
)

processor = AutoProcessor.from_pretrained(MODEL_ID)
processor.tokenizer.padding_side = "right"

## fine-tuning setup
# peft_config = LoraConfig(
#     lora_alpha=16,
#     lora_dropout=0.1,
#     r=8,
#     bias="none",
#     target_modules=["q_proj", "v_proj"],
#     task_type="CAUSAL_LM",
# )

# print(f"Before adapter parameters: {model.num_parameters()}")
# peft_model = get_peft_model(model, peft_config)
# peft_model.print_trainable_parameters()

# training_args = SFTConfig(
#     output_dir="/kaggle/working/",
#     num_train_epochs=EPOCHS,
#     per_device_train_batch_size=BATCH_SIZE,
#     per_device_eval_batch_size=BATCH_SIZE,
#     gradient_checkpointing=GRADIENT_CHECKPOINTING,
#     learning_rate=LEARNING_RATE,
#     logging_steps=LOGGING_STEPS,
#     eval_steps=EVAL_STEPS,
#     eval_strategy=EVAL_STRATEGY,
#     save_strategy=SAVE_STRATEGY,
#     save_steps=SAVE_STEPS,
#     metric_for_best_model=METRIC_FOR_BEST_MODEL,
#     load_best_model_at_end=LOAD_BEST_MODEL_AT_END,
#     max_grad_norm=MAX_GRAD_NORM,
#     warmup_steps=WARMUP_STEPS,
#     dataset_kwargs=DATASET_KWARGS,
#     max_seq_length=MAX_SEQ_LEN,
#     remove_unused_columns = REMOVE_UNUSED_COLUMNS,
#     optim=OPTIM,
# )

# collate_sample = [train_dataset[0], train_dataset[1]] # for batch size 2.

# def collate_fn(examples):
#     texts = [processor.apply_chat_template(example, tokenize=False) for example in examples]
#     image_inputs = [example[1]["content"][0]["image"] for example in examples]

#     batch = processor(
#         text=texts, images=image_inputs, return_tensors="pt", padding=True
#     )
#     labels = batch["input_ids"].clone()
#     labels[labels == processor.tokenizer.pad_token_id] = -100
#     batch["labels"] = batch["input_ids"]

#     return batch

# collated_data = collate_fn(collate_sample)
# print(collated_data.keys())

# trainer = SFTTrainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=eval_dataset,
#     data_collator=collate_fn,
#     peft_config=peft_config,
#     processing_class=processor.tokenizer,
# )

## load fine-tuned checkpoints
model.load_adapter("model_weight/qwen-2-5-vl-3b-checkpoint-1350")


## inference functions
def text_generator(sample_data):
    text = processor.apply_chat_template(
        sample_data["messages"][0:2], tokenize=False, add_generation_prompt=True
    )

    print(f"Prompt: {text}")
    print("-"*30)

    image_inputs =  sample_data["messages"][1]["content"][0]["image"]

    inputs = processor(
        text=[text],
        images = image_inputs,
        return_tensors="pt"
    )
    inputs = inputs.to(device)

    generated_ids = model.generate(**inputs, max_new_tokens=MAX_SEQ_LEN)

    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True
    )
    del inputs
    actual_answer = sample_data["messages"][2]["content"][0]["text"]
    return output_text[0], actual_answer


def run_inference(dataset):
    results = []

    for samples in dataset:
    
        try:
            predicted, actual = text_generator(samples)
            print(samples['filename'])
            # print("predictions++++++++",predicted)
            
            results.append({
                "image_file": samples['filename'],
                "predicted": predicted,
                "actual": actual
            })
        except Exception as e:
            print(f"Error at file: {e}")

    # Save to JSON
    with open("finetuned_qwen_inference_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Inference completed and saved to inference_results.json")    

run_inference(eval_dataset)