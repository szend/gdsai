from fastapi import FastAPI
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

MODEL_NAME = "t5-small" 
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, use_fast=True)

model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

app = FastAPI()

@app.get("/input")
async def read_input(inputstr: str = ""): 
    
    inputs = tokenizer(inputstr, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(**inputs, max_length=256, num_beams=1)
    return {"generated_text": generated_text}  
