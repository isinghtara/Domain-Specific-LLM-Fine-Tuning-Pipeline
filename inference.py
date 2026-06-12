import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "mistralai/Mistral-7B-v0.1"
ADAPTER_PATH = "./finetuned_model"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    ADAPTER_PATH
)

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load LoRA adapter
model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH
)

prompt = """
### Instruction:
What are symptoms of diabetes?

### Response:
"""

inputs = tokenizer(
    prompt,
    return_tensors="pt"
).to(model.device)

output = model.generate(
    **inputs,
    max_new_tokens=150,
    temperature=0.7,
    do_sample=True
)

print(
    tokenizer.decode(
        output[0],
        skip_special_tokens=True
    )
)