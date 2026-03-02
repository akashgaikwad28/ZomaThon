from fastapi import Request, status
from fastapi.responses import JSONResponse
from app.core.logger import logger

class ZomaThonError(Exception):
    """Base exception for the ZomaThon project."""
    def __init__(self, message: str, status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)

class RecommendationError(ZomaThonError):
    """Raised when the recommendation pipeline (retrieval/ranking) fails."""
    def __init__(self, message: str = "Failed to generate recommendations"):
        super().__init__(message, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

class FeatureStoreError(ZomaThonError):
    """Raised when feature retrieval or processing fails."""
    def __init__(self, message: str = "Failed to fetch necessary features from the feature store"):
        super().__init__(message, status_code=status.HTTP_503_SERVICE_UNAVAILABLE)

async def zomathon_exception_handler(request: Request, exc: ZomaThonError) -> JSONResponse:
    """Global handler for expected custom ZomaThon exceptions."""
    # Log the exact error along with the path it occurred on
    logger.error(f"ZomaThonError at {request.url.path}: {exc.message}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content={"success": False, "error": exc.message}
    )

async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Global fallback handler for unhandled exceptions to prevent crashing and leak of stack traces."""
    logger.error(f"Unhandled Exception at {request.url.path}: {str(exc)}")
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"success": False, "error": "Internal Server Error"}
    )
