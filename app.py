from fastapi import FastAPI
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch



app = FastAPI()

@app.get("/input")
async def read_input(inputstr: str = ""): 
    

    return {"generated_text": "1"}  
