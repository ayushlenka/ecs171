from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from routes import predict, sentiment

app = FastAPI(title="Stock Sentiment Analysis API")

# ✅ CORS Middleware (Keep this)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Use "*" only for development
    allow_credentials=True,
    allow_methods=["*"],  # ✅ Allow all methods
    allow_headers=["*"],  # ✅ Allow all headers
)

# ✅ Workaround: Handle OPTIONS Requests Manually
@app.options("/{full_path:path}")
async def preflight_handler(full_path: str):
    """Manually respond to CORS preflight (OPTIONS) requests."""
    return JSONResponse(
        content={},  # Empty response body
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "http://localhost:3000",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization",
        },
    )

# ✅ Include API routes
app.include_router(predict.router)
app.include_router(sentiment.router)

@app.get("/")
def home():
    return {"message": "Stock Sentiment Analysis API is running!"}
