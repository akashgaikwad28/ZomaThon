import redis
import json
from app.core.config import settings
from app.core.logger import logger
from typing import List, Dict, Any, Optional

class RedisClient:
    def __init__(self):
        try:
            redis_kwargs = {
                "host": settings.REDIS_HOST,
                "port": settings.REDIS_PORT,
                "decode_responses": True
            }
            if settings.REDIS_PASSWORD:
                redis_kwargs["password"] = settings.REDIS_PASSWORD
                
            self.client = redis.Redis(**redis_kwargs)
            # Test connection
            self.client.ping()
            logger.info(f"Connected to Redis at {settings.REDIS_HOST}:{settings.REDIS_PORT}")
            self.enabled = True
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}. Falling back to in-memory/DB.")
            self.enabled = False

    def get_candidates(self, item_id: int) -> Optional[List[int]]:
        if not self.enabled:
            return None
        
        try:
            val = self.client.get(f"cand:{item_id}")
            if val:
                return json.loads(val)
            return None
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            return None

    def set_candidates(self, item_id: int, candidates: List[int], ttl: int = 3600*24):
        if not self.enabled:
            return
            
        try:
            self.client.setex(f"cand:{item_id}", ttl, json.dumps(candidates))
        except Exception as e:
            logger.error(f"Redis set error: {e}")

    def get_explanation(self, item_id: int, category: str) -> list:
        if not self.client: return None
        try:
            return self.client.get(f"expl:{item_id}:{category}")
        except Exception as e:
            logger.error(f"Redis get_explanation error: {e}")
            return None

    def set_explanation(self, item_id: int, category: str, explanation: str):
        if not self.client: return
        try:
            self.client.set(f"expl:{item_id}:{category}", explanation, ex=86400) # 24 hrs
        except Exception as e:
            logger.error(f"Redis set_explanation error: {e}")

redis_client = RedisClient()
