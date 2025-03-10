from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import logging
from models.llm import prompting  # Ensure this import is correct

router = APIRouter()

class TweetRequest(BaseModel):
    tweet: str

@router.post("/analyze_tweet")
async def analyze_tweet(request: TweetRequest):
    try:
        logging.info(f"Received tweet: {request.tweet}")

        # Call LLM function
        result = prompting(request.tweet)

        logging.info(f"LLM response: {result}")  # ✅ Log what LLM returns

        if not result or not isinstance(result, str):
            raise HTTPException(status_code=500, detail="LLM response is invalid")

        # Ensure result format is correct before splitting
        parts = result.split()
        if len(parts) < 2:
            raise HTTPException(status_code=500, detail=f"Unexpected LLM output: {result}")

        emotion, market_trend = parts[:2]  # Only take the first two parts

        return {
            "emotion": emotion,
            "sentiment": market_trend
        }

    except Exception as e:
        logging.error(f"Error processing tweet: {e}")
        raise HTTPException(status_code=500, detail=str(e))
