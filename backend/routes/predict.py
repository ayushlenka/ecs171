import torch
import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
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
    sentiment_score: float  # Float

class StockPredictionModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(StockPredictionModel, self).__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])  # Get last time step output

def load_model(stock_name):
    model_path = os.path.join(MODEL_DIR, STOCK_MODELS.get(stock_name))

    if not os.path.exists(model_path):
        raise HTTPException(status_code=500, detail=f"Model not found for {stock_name}")

    try:
        model = StockPredictionModel(input_size=13, hidden_size=32, output_size=1, num_layers=2)

        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")), strict=False)
        model.eval()
        return model
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

@router.post("/predict")
async def predict_stock(request: PredictionRequest):
    try:
        logging.info(f"📩 Received prediction request: {request}")

        if request.company not in STOCK_MODELS:
            logging.error(f"Invalid company: {request.company}")
            raise HTTPException(status_code=400, detail="Invalid company. Choose from: Amazon, Apple, Tesla, Microsoft")

        model = load_model(request.company)

        input_features = [request.sentiment_score] + [0.0] * 12  
        input_tensor = torch.tensor(input_features, dtype=torch.float32)

        input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)

        logging.info(f"Model input tensor shape: {input_tensor.shape}") 

        predicted_change = model(input_tensor).item()

        adjusted_change = round(predicted_change * request.sentiment_score, 4)

        if adjusted_change > 0:
            trend = "Positive"
        elif adjusted_change < 0:
            trend = "Negative"
        else:
            trend = "Stable"

        logging.info(f"Model output for {request.company}: {adjusted_change} ({trend})")

        return {
            "company": request.company,
            "predicted_change": adjusted_change,
            "trend": trend
        }

    except Exception as e:
        logging.error(f"Error predicting stock: {e}")
        raise HTTPException(status_code=500, detail=str(e))
