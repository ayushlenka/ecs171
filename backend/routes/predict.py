from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import torch
import os
import logging

router = APIRouter()

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../models/parameters")

STOCK_MODELS = {
    "Amazon": "AMZN.pth",
    "Apple": "AAPL.pth",
    "Tesla": "TSLA.pth",
    "Microsoft": "MSFT.pth"
}

class PredictionRequest(BaseModel):
    company: str
    sentiment_score: float  # ✅ Ensure this is a float

def load_model(stock_name):
    model_path = os.path.join(MODEL_DIR, STOCK_MODELS.get(stock_name))

    if not os.path.exists(model_path):
        raise HTTPException(status_code=500, detail=f"Model not found for {stock_name}")

    model = torch.jit.load(model_path, map_location=torch.device("cpu"))
    model.eval()
    return model

@router.post("/predict")
async def predict_stock(request: PredictionRequest):
    try:
        logging.info(f"Received prediction request: {request}")

        if request.company not in STOCK_MODELS:
            raise HTTPException(status_code=400, detail="Invalid company. Choose from: Amazon, Apple, Tesla, Microsoft")

        model = load_model(request.company)

        # Convert sentiment score to tensor input
        input_tensor = torch.tensor([[request.sentiment_score]], dtype=torch.float32)

        # Make prediction
        predicted_change = model(input_tensor).item()

        return {
            "company": request.company,
            "predicted_change": predicted_change
        }

    except Exception as e:
        logging.error(f"Error predicting stock: {e}")
        raise HTTPException(status_code=500, detail=str(e))
