from fastapi import FastAPI

app = FastAPI(title="Video Analysis API", version="0.1.0")


@app.get("/health")
def health():
    return {"status": "ok"}