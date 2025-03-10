from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from models.llm import prompting  # Import LLM function

router = APIRouter()

# Define input model
class SentimentRequest(BaseModel):
    tweet: str

@router.post("/analyze-sentiment")
async def analyze_sentiment(request: SentimentRequest):
    try:
        result = prompting(request.tweet)  # Call LLM function
        emotion, market_trend = result.split()

        return {
            "emotion": emotion,
            "market_trend": market_trend
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
