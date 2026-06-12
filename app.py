from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Request(BaseModel):
    prompt: str

@app.post("/generate")
async def generate(req: Request):

    inputs = tokenizer(
        req.prompt,
        return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=200
    )

    response = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return {"response": response}