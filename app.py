from fastapi import FastAPI

from transformers import T5Tokenizer, T5ForConditionalGeneration



app = FastAPI()


MODEL_NAME = "t5-small" 
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

@app.get("/")
async def read_root(name: str = ""): 
    
    inputs = tokenizer(str, return_tensors="pt", padding=True, truncation=True)

    outputs = model.generate(**inputs, max_length=256, num_beams=2, early_stopping=True)

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"generated_text": generated_text}
