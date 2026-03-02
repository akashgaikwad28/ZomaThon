from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

from app.services.recommendation_service import RecommendationService
from app.core.logger import logger

router = APIRouter()
rec_service = RecommendationService()

class RecommendRequest(BaseModel):
    user_id: int
    restaurant_id: int
    cart_items: List[int]
    timestamp: Optional[str] = None

class Recommendation(BaseModel):
    item_id: int
    score: float
    explanation: Optional[str] = None

class RecommendResponse(BaseModel):
    recommendations: List[Recommendation]

@router.post("/recommend", response_model=RecommendResponse)
async def get_recommendations(request: RecommendRequest, background_tasks: BackgroundTasks):
    """
    Main recommendation endpoint.
    Retrieves candidates, ranks them using LightGBM, and applies business rules.
    """
    try:
        logger.info(f"Received recommendation request for user {request.user_id}")
        
        results = rec_service.get_recommendations(
            user_id=request.user_id,
            restaurant_id=request.restaurant_id,
            cart_items=request.cart_items
        )
        
        final_response = []
        for r in results:
            if "_needs_llm" in r:
                task_data = r.pop("_needs_llm")
                from app.services.llm_service import llm_service
                background_tasks.add_task(
                    llm_service.generate_and_cache_explanation,
                    task_data["item_id"],
                    task_data["category"],
                    task_data["item_name"],
                    task_data["cart_names"]
                )
            
            final_response.append(
                Recommendation(
                    item_id=r["item_id"], 
                    score=r["score"],
                    explanation=r.get("explanation")
                )
            )
            
        return RecommendResponse(recommendations=final_response)
    except Exception as e:
        logger.error(f"Error processing recommendation: {e}")
        raise HTTPException(status_code=500, detail="Internal server error in recommendation pipeline")
