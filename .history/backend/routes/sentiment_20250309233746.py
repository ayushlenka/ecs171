from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from fastapi.responses import JSONResponse

router = APIRouter()

class TweetRequest(BaseModel):
    tweet: str

@router.options("/analyze_tweet")
async def options_analyze_tweet():
    """Explicitly handle OPTIONS requests for CORS."""
    return JSONResponse(status_code=200, headers={"Access-Control-Allow-Origin": "*", "Access-Control-Allow-Methods": "POST, OPTIONS", "Access-Control-Allow-Headers": "Content-Type"})

@router.post("/analyze_tweet")
async def analyze_tweet(request: TweetRequest):
    """Handles tweet analysis."""
    try:
        return {"emotion": "Excitement", "sentiment": "bullish"}  # Dummy response for testing
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
