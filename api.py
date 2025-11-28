from fastapi import FastAPI
from hybrid_parser import hybrid_parse

app = FastAPI()

@app.get("/")
def home():
    return {"status": "running"}

@app.post("/parse")
def parse_sms(data: dict):
    return hybrid_parse(data["sms"])
