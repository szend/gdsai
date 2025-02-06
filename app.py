from fastapi import FastAPI
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

MODEL_NAME = "google/t5-v1_1-tiny" 
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME,use_fast=True, token="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")

model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

app = FastAPI()

@app.get("/input")
async def read_input(inputstr: str = ""): 
    
    inputs = tokenizer(inputstr, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(**inputs, max_length=256, num_beams=1)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return {"generated_text": generated_text}  
