from fastapi import FastAPI
from routes import predict, sentiment
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow frontend to communicate with FastAPI backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with frontend URL in production
    allow_credentials=True,
    allow_methods=["OPTIONS", "POST"],  # Allow OPTIONS and POST
    allow_headers=["*"],  # Allow all headers
)


app = FastAPI(title="Stock Sentiment Analysis API")

# Include API routes
app.include_router(predict.router)
app.include_router(sentiment.router)

@app.get("/")
def home():
    return {"message": "Stock Sentiment Analysis API is running!"}
