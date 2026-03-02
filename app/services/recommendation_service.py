import os
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any

from app.core.logger import logger
from app.services.business_rules import BusinessReRanker, compute_intra_list_diversity
from app.features.cart_stage import detect_cart_stage

class RecommendationService:
    """
    Orchestration layer that performs retrieval, ranking, and re-ranking.
    """
    
    def __init__(self):
        self.model_dir = "app/models/weights"
        self.processed_dir = "data/processed"
        self.re_ranker = BusinessReRanker()
        self.load_artifacts()

    def load_artifacts(self):
        """Loads model weights, retrieval indices, and features into memory."""
        try:
            logger.info("Loading recommendation artifacts into memory...")
            with open(f"{self.model_dir}/co_occurrence.pkl", "rb") as f:
                self.co_occurrence_matrix = pickle.load(f)
            
            with open(f"{self.model_dir}/ranker_model.pkl", "rb") as f:
                self.ranker_model = pickle.load(f)
                
            try:
                with open(f"{self.model_dir}/item_embeddings.pkl", "rb") as f:
                    self.item_embeddings = pickle.load(f)
            except FileNotFoundError:
                self.item_embeddings = {}
                
            if os.path.exists(f"{self.processed_dir}/user_features.parquet"):
                self.user_features = pd.read_parquet(f"{self.processed_dir}/user_features.parquet").set_index("user_id")
            else:
                self.user_features = pd.DataFrame()
                
            if os.path.exists(f"{self.processed_dir}/item_features.parquet"):
                self.item_features = pd.read_parquet(f"{self.processed_dir}/item_features.parquet").set_index("item_id")
            else:
                self.item_features = pd.DataFrame()
                
            if os.path.exists(f"{self.processed_dir}/restaurant_features.parquet"):
                self.restaurant_features = pd.read_parquet(f"{self.processed_dir}/restaurant_features.parquet").set_index("restaurant_id")
            else:
                self.restaurant_features = pd.DataFrame()
            
            logger.info("Artifacts loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load artifacts: {e}")
            self.co_occurrence_matrix = {}
            self.ranker_model = None
            self.ranker_model = None
            self.user_features = pd.DataFrame()
            self.item_features = pd.DataFrame()
            self.restaurant_features = pd.DataFrame()
            self.item_embeddings = {}

    def get_recommendations(self, user_id: int, restaurant_id: int, cart_items: List[int]) -> List[Dict[str, Any]]:
        """
        Full 3-layer pipeline execution.
        """
        # 1. Candidate Retrieval (Layer 1) using Redis Cache for sub-30ms latency
        candidates = set()
        from app.core.redis import redis_client
        
        for item_id in cart_items:
            # Try Redis first
            cached_cands = redis_client.get_candidates(item_id)
            if cached_cands:
                candidates.update(cached_cands)
            else:
                # Fallback to in-memory matrix if Redis is unavailable/miss
                neighbors = self.co_occurrence_matrix.get(item_id, [])
                candidates.update(neighbors)
                # Populate cache for future requests
                if neighbors:
                    redis_client.set_candidates(item_id, neighbors)
            
        # Fallback: Advanced Cold Start Strategy (Phase 6)
        if len(candidates) < 5:
            from app.services.cold_start import get_cold_start_recommendations
            fallback_cands = get_cold_start_recommendations(
                user_id, restaurant_id, cart_items, 
                self.user_features, self.item_features, getattr(self, "restaurant_features", pd.DataFrame())
            )
            candidates.update(fallback_cands)
            
        # 1.5 Detect Cart Stage (Phase 6)
        cart_stage = detect_cart_stage(cart_items, self.item_features)
        
        # 2. Ranking (Layer 2)
        # Prepare feature vectors for candidates
        user_feat = self.user_features.loc[[user_id]] if user_id in self.user_features.index else None
        
        # Calculate cart embedding (mean of item embeddings)
        cart_vecs = [self.item_embeddings[item_id] for item_id in cart_items if item_id in getattr(self, "item_embeddings", {})]
        if cart_vecs:
            cart_embedding = np.mean(cart_vecs, axis=0)
            cart_norm = np.linalg.norm(cart_embedding)
        else:
            cart_embedding = None
            cart_norm = 0.0
            
        # Fetch restaurant context
        rest_feat = self.restaurant_features.loc[[restaurant_id]] if getattr(self, "restaurant_features", None) is not None and restaurant_id in getattr(self, "restaurant_features", pd.DataFrame()).index else None
        rest_pop = rest_feat["restaurant_popularity_score"].values[0] if rest_feat is not None and "restaurant_popularity_score" in rest_feat else 0.0

        ranking_data = []
        for cand_id in candidates:
            if cand_id not in self.item_features.index: continue
            
            item_feat = self.item_features.loc[cand_id]
            
            hour = datetime.now().hour
            is_veg_user = user_feat["is_veg_user"].values[0] if user_feat is not None and "is_veg_user" in user_feat else 0
            avg_order_value = user_feat["avg_order_value"].values[0] if user_feat is not None else 500
            
            day_of_week = datetime.now().weekday()
            
            # Build feature row (must match training features)
            feature_row = {
                "is_veg_user": is_veg_user,
                "avg_order_value": avg_order_value,
                "order_count": user_feat["order_count"].values[0] if user_feat is not None else 0,
                "price": float(item_feat["price"]),
                "is_veg_item": int(item_feat["is_veg_item"]) if "is_veg_item" in item_feat else 0,
                "popularity_score": item_feat["popularity_score"],
                "price_deviation_from_user_avg": float(item_feat["price"] - avg_order_value),
                "veg_alignment_score": 1 if is_veg_user == (item_feat["is_veg_item"] if "is_veg_item" in item_feat else 0) else 0,
                "user_item_interaction_count": 0, 
                "restaurant_popularity_score": rest_pop,
                
                # Phase 7 Features
                "recency_days": float(user_feat["recency_days"].values[0]) if user_feat is not None and "recency_days" in user_feat else 999.0,
                "frequency_per_month": float(user_feat["frequency_per_month"].values[0]) if user_feat is not None and "frequency_per_month" in user_feat else 0.0,
                "dessert_add_rate": float(user_feat["dessert_add_rate"].values[0]) if user_feat is not None and "dessert_add_rate" in user_feat else 0.0,
                "beverage_add_rate": float(user_feat["beverage_add_rate"].values[0]) if user_feat is not None and "beverage_add_rate" in user_feat else 0.0,
                "price_elasticity_score": float(user_feat["price_elasticity_score"].values[0]) if user_feat is not None and "price_elasticity_score" in user_feat else 1.0,
                "popularity_percentile": float(item_feat["popularity_percentile"]) if "popularity_percentile" in item_feat else 0.5,
                "seasonal_score": float(item_feat["seasonal_score"]) if "seasonal_score" in item_feat else 1.0,
                "margin_estimate": float(item_feat["margin_estimate"]) if "margin_estimate" in item_feat else float(item_feat["price"])*0.2,
                "hour_bucket": hour // 4,
                "weekend_flag": 1 if day_of_week >= 5 else 0,
                "meal_type": 1 if 12<=hour<=15 else (2 if 19<=hour<=22 else 0),
                "city_embedding": 1.0 if user_feat is not None and "city_user" in user_feat and user_feat["city_user"].values[0] == "Mumbai" else 0.0,
                
                # Dynamic Re-Ranker State (Calculated later, not fed to initial ML)
                "meal_completion_score": cart_stage.get("meal_completion_score", 0.0),
                "cart_embedding_similarity": 0.0
            }
            
            # Calculate Cosine Similarity with cart_embedding
            if cart_embedding is not None and cand_id in getattr(self, 'item_embeddings', {}):
                cand_vec = self.item_embeddings[cand_id]
                cand_norm = np.linalg.norm(cand_vec)
                if cart_norm > 0 and cand_norm > 0:
                    sim = np.dot(cart_embedding, cand_vec) / (cart_norm * cand_norm)
                    feature_row["cart_embedding_similarity"] = float(sim)
                    
            ranking_data.append({"item_id": cand_id, "features": feature_row, "category": item_feat["category"]})
            
        if not ranking_data:
            return []
            
        # Batch inference
        features_df = pd.DataFrame([rd["features"] for rd in ranking_data])
        
        ml_features = [
            "is_veg_user", "avg_order_value", "order_count",
            "price", "is_veg_item", "popularity_score",
            "price_deviation_from_user_avg", "veg_alignment_score",
            "user_item_interaction_count", "restaurant_popularity_score",
            "recency_days", "frequency_per_month", "dessert_add_rate", 
            "beverage_add_rate", "price_elasticity_score", "popularity_percentile",
            "seasonal_score", "margin_estimate", "hour_bucket", "weekend_flag",
            "meal_type", "city_embedding"
        ]
        
        scores = self.ranker_model.predict_proba(features_df[ml_features])[:, 1] if self.ranker_model else [0.0] * len(ranking_data)
        
        scored_candidates = []
        for rd, score in zip(ranking_data, scores):
            scored_candidates.append({
                "item_id": rd["item_id"],
                "score": float(score),
                "category": rd["category"]
            })
            
        # Sort by score
        scored_candidates = sorted(scored_candidates, key=lambda x: x["score"], reverse=True)
            
        # 3. Business Re-ranking (Layer 3)
        final_recs = self.re_ranker.re_rank(scored_candidates, cart_items, cart_stage)
        
        # Log Diversity
        diversity_score = compute_intra_list_diversity(final_recs)
        logger.info(f"Generated recommendations with Intra-List Diversity Score: {diversity_score:.2f}")

        # 4. LLM Explanations (Phase 5 -> Phase 6 Async)
        # We fetch real item names to send to LLM.
        try:
            cart_names = [self.item_features.loc[c]["item_name"] for c in cart_items if c in self.item_features.index]
        except Exception:
            cart_names = []
            
        for rec in final_recs:
            if final_recs.index(rec) == 0 and cart_names:
                item_id = rec["item_id"]
                category = rec.get("category", "")
                
                # Check Redis Cache First (Phase 6)
                from app.core.redis import redis_client
                cached_expl = redis_client.get_explanation(item_id, category)
                
                if cached_expl:
                    rec["explanation"] = cached_expl
                else:
                    # Default instantaneous fallback
                    rec["explanation"] = "Pairs perfectly with your current meal."
                    # Flag for the API layer to fire a BackgroundTask
                    rec["_needs_llm"] = {
                        "item_id": item_id,
                        "category": category,
                        "item_name": self.item_features.loc[item_id]["item_name"] if item_id in getattr(self, "item_features", pd.DataFrame()).index else "an item",
                        "cart_names": cart_names
                    }
            else:
                rec["explanation"] = "Pairs perfectly with your current meal."
        
        return final_recs
