from sqlalchemy import Column, Integer, String, Float, Boolean, JSON
from app.db.session import Base

class UserFeature(Base):
    __tablename__ = "user_features"
    
    user_id = Column(Integer, primary_key=True, index=True)
    city = Column(String, index=True)
    is_veg = Column(Integer)
    avg_order_value = Column(Float)
    order_count = Column(Integer)

class ItemFeature(Base):
    __tablename__ = "item_features"
    
    item_id = Column(Integer, primary_key=True, index=True)
    restaurant_id = Column(Integer, index=True)
    item_name = Column(String)
    category = Column(String, index=True)
    price = Column(Float)
    is_veg = Column(Integer)
    is_addon = Column(Boolean)
    popularity_score = Column(Float)

class CandidateCache(Base):
    """
    Fallback table if Redis is unavailable.
    Stores precomputed item-item candidates.
    """
    __tablename__ = "candidate_cache"
    
    item_id = Column(Integer, primary_key=True, index=True)
    candidates_json = Column(JSON) # List of candidate item IDs
