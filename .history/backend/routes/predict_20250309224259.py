from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import torch
import os
import numpy as np

# Define API router
router = APIRouter()

# Define directory for model parameters
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../models/parameters")

# Define stock models mapping
STOCK_MODELS = {
    "Amazon": "AMZN.pth",
    "Apple": "AAPL.pth",
    "Tesla": "TSLA.pth",
    "Microsoft": "MSFT.pth"
}

# Define input model structure
class PredictionRequest(BaseModel):
    company: str
    sentiment_score: float  # Sentiment score from llm.py

# Function to load model
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
        if request.company not in STOCK_MODELS:
            raise HTTPException(status_code=400, detail="Invalid company. Choose from: Amazon, Apple, Tesla, Microsoft")

        model = load_model(request.company)

        # Prepare input tensor (simulate input format based on sentiment)
        input_tensor = torch.tensor([[request.sentiment_score]], dtype=torch.float32)

        # Make prediction
        predicted_change = model(input_tensor).item()

        return {
            "company": request.company,
            "predicted_change": predicted_change
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
