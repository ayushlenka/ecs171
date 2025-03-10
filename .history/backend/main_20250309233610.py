from fastapi import FastAPI
from routes import predict, sentiment
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow frontend to communicate with FastAPI backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this later to restrict access
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],  # Allow all headers
)


app = FastAPI(title="Stock Sentiment Analysis API")

# Include API routes
app.include_router(predict.router)
app.include_router(sentiment.router)

@app.get("/")
def home():
    return {"message": "Stock Sentiment Analysis API is running!"}
