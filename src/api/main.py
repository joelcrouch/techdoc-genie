from fastapi import FastAPI

app = FastAPI(title="Technical Documentation Assistant API")

@app.get("/")
def read_root():
    return {"status": "API is running"}
