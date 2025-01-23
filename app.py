from fastapi import FastAPI

import logging



app = FastAPI()



# Konfiguráld a naplózást

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("uvicorn.error")



@app.get("/")

async def root():

    logger.info("Root endpoint called")

    return {"message": "Hello, World"}



@app.get("/error")

async def cause_error():

    logger.error("Something went wrong!")

    raise RuntimeError("This is a test error")
