import time
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.core.logger import logger
from app.core.exceptions import (
    ZomaThonError, 
    zomathon_exception_handler, 
    global_exception_handler
)

from app.api.endpoints import router as api_router

# Initialize FastAPI application
app = FastAPI(
    title="ZomaThon Recommendation API",
    description="Modular, scalable, real-time recommendation microservice",
    version="1.0.0"
)

# Register Custom Exception Handlers
app.add_exception_handler(ZomaThonError, zomathon_exception_handler)
app.add_exception_handler(Exception, global_exception_handler)

# Register Routers
app.include_router(api_router, prefix="/api/v1")

@app.middleware("http")
async def log_latency_middleware(request: Request, call_next):
    """
    Middleware to calculate and log the latency of every incoming request.
    Essential for real-time latency monitoring (<250ms SLA).
    """
    start_time = time.time()
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    latency_ms = process_time * 1000
    
    # Log the access details and latency
    logger.info(
        f"Path: {request.url.path} | Method: {request.method} | "
        f"Status: {response.status_code} | Latency: {latency_ms:.2f}ms"
    )
    
    # Optionally append a custom header for debugging
    response.headers["X-Process-Time"] = str(process_time)
    return response

@app.get("/health", tags=["System"])
async def health_check():
    """Simple health check endpoint for monitoring."""
    return {"status": "healthy", "service": "ZomaThon Recommendation API"}

if __name__ == "__main__":
    import uvicorn
    # Local dev entrypoint
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
