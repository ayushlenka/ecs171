from fastapi import FastAPI
from routes import predict, sentiment

app = FastAPI(title="Stock Sentiment Analysis API")

# Include API routes
app.include_router(predict.router)
app.include_router(sentiment.router)

@app.get("/")
def home():
    return {"message": "Stock Sentiment Analysis API is running!"}
