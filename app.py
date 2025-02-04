from fastapi import FastAPI
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch  
from concurrent.futures import ThreadPoolExecutor  


MODEL_NAME = "t5-small" 
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, use_fast=True)

model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

app = FastAPI()

executor = ThreadPoolExecutor(max_workers=4)

async def generate_text(inputstr):  
    inputs = tokenizer(inputstr, return_tensors="pt", padding=True, truncation=True)  

    # Use mixed precision if using CUDA and supported  
    with torch.no_grad():  
        outputs = model.generate(**inputs, max_length=256, num_beams=1, early_stopping=True)  

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)  
    return generated_text  

@app.get("/input")  
async def read_input(inputstr: str = ""):   
    generated_text = await asyncio.get_event_loop().run_in_executor(executor, generate_text, inputstr)  
    return {"generated_text": generated_text}  
