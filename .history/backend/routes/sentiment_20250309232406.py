from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from models.llm import prompting  # Make sure this imports correctly

router = APIRouter()

class TweetRequest(BaseModel):
    tweet: str

@router.post("/analyze_tweet")
async def analyze_tweet(request: TweetRequest):
    try:
        result = prompting(request.tweet)
        emotion, sentiment = result.split()  # Split into emotion and sentiment
        
        return {
            "emotion": emotion,
            "sentiment": sentiment
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
