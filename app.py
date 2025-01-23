from fastapi import FastAPI

from pydantic import BaseModel

from transformers import T5Tokenizer, T5ForConditionalGeneration



app = FastAPI()


MODEL_NAME = ".\\myt5small"

tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)


class InputData(BaseModel):

    input_text: str



@app.get("/")

def root():

    return {"message": "T5 API is running!"}


@app.post("/generate/")

def generate_text(data: InputData):


    inputs = tokenizer(data.input_text, return_tensors="pt", padding=True, truncation=True)

 

    outputs = model.generate(**inputs, max_length=50, num_beams=5, early_stopping=True)


    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    

    return {"generated_text": generated_text}





