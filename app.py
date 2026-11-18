import io
import os

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

app = FastAPI(title="Image classifier")

# Load a small pretrained model once at startup (weights download on first run)
weights = MobileNet_V3_Small_Weights.DEFAULT
model = mobilenet_v3_small(weights=weights)
model.eval()
preprocess = weights.transforms()
labels = weights.meta["categories"]

# Optional database: only used when DATABASE_UR is set (will do later)
DATABASE_URL = os.getenv("DATABASE_URL")

def save_prediction(filename: str, label: str, probability: float) -> None:
    if not DATABASE_URL:
        return
    import psycopg

    with psycopg.connect(DATABASE_URL) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS predictions ("
            " id SERIAL PRIMARY KEY, filename TEXT, label TEXT,"
            " probability REAL, created_at TIMESTAMPTZ DEFAULT now())"
        )
        conn.execute(
            "INSERT INTO predictions (filename, label, probability) VALUES (%s, %s, %s)",
            (filename, label, probability),
        )




@app.get("/health")
def health():
    return {"status": "ok", "torch": torch.__version__, "database": bool(DATABASE_URL)}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="File must have a name")
    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Upload a JPEG or PNG image")
    batch = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        probs = model(batch).softmax(dim=1)[0]
    top = probs.topk(3)
    predictions = [
        {"label": labels[i], "probability": round(p.item(), 4)}
        for p, i in zip(top.values, top.indices)
    ]
    save_prediction(file.filename, predictions[0]["label"], predictions[0]["probability"])
    return {"filename": file.filename, "predictions": predictions}


@app.get("/history")
def history():
    if not DATABASE_URL:
        return {"detail": "No database configured", "predictions": []}
    import psycopg

    with psycopg.connect(DATABASE_URL) as conn:
        rows = conn.execute(
            "SELECT filename, label, probability, created_at FROM predictions"
            " ORDER BY id DESC LIMIT 20"
        ).fetchall()
    return {
        "predictions": [
            {"filename": f, "label": l, "probability": p, "created_at": c.isoformat()}
            for f, l, p, c in rows
        ]
    }