from infer import inference
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import json

app = FastAPI()

class payload(BaseModel):
    sentence: str


@app.get("/health")
async def health_check():
    return "<h>Im Alive!</h>"

@app.post("/inference")
async def sentiment(item: payload):
    result : dict = inference(item.sentence)
    return result


if __name__ == "__main__":
    uvicorn.run(app, port=8080, host='0.0.0.0')