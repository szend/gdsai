from fastapi import FastAPI
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

MODEL_NAME = "google/t5-efficient-mini" 
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME,use_fast=True, token="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")


app = FastAPI()

@app.get("/")
async def read_input(inputstr: str = "alma"): 
    inputs = tokenizer(inputstr, return_tensors="pt", padding=True, truncation=True)
    torch.save(inputs, "tokenized_inputs.pt")  
    loaded_inputs = torch.load("tokenized_inputs.pt")  



    return {loaded_inputs}  


@app.get("/input")
async def read_input(inputstr: str = "alma"): 
    inputs = tokenizer(inputstr, return_tensors="pt", padding=True, truncation=True)




    return {inputs}  
