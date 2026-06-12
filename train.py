import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model
)
from trl import SFTTrainer

# ------------------------------------
# Configuration
# ------------------------------------

MODEL_NAME = "mistralai/Mistral-7B-v0.1"
DATASET_PATH = "data/train.json"

# ------------------------------------
# Load Dataset
# ------------------------------------

dataset = load_dataset(
    "json",
    data_files=DATASET_PATH,
    split="train"
)

# Convert to instruction format
def format_prompt(example):
    return {
        "text": f"""
### Instruction:
{example['instruction']}

### Response:
{example['output']}
"""
    }

dataset = dataset.map(format_prompt)

# ------------------------------------
# Quantization (QLoRA)
# ------------------------------------

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

# ------------------------------------
# Tokenizer
# ------------------------------------

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

tokenizer.pad_token = tokenizer.eos_token

# ------------------------------------
# Base Model
# ------------------------------------

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto"
)

# ------------------------------------
# LoRA Configuration
# ------------------------------------

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

model.print_trainable_parameters()

# ------------------------------------
# Training Arguments
# ------------------------------------

training_args = TrainingArguments(
    output_dir="./finetuned_model",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="epoch",
    fp16=True,
    optim="paged_adamw_8bit",
    report_to="none"
)

# ------------------------------------
# Trainer
# ------------------------------------

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args
)

# ------------------------------------
# Start Training
# ------------------------------------

trainer.train()

# Save Model
trainer.model.save_pretrained(
    "./finetuned_model"
)

tokenizer.save_pretrained(
    "./finetuned_model"
)

print("Training Completed!")