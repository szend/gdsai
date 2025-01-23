from fastapi import FastAPI

from pydantic import BaseModel


app = FastAPI()


class InputData(BaseModel):

    input_text: str



@app.get("/")

def root():

    return {"message": "T5 API is running!"}




