from fastapi import FastAPI




app = FastAPI()

@app.get("/")

def root():

    return {"message": "T5 API is running!"}




