import pandas as pd
from typing import List, Dict, Any

def get_cold_start_recommendations(
    user_id: int, 
    restaurant_id: int, 
    cart_items: List[int],
    user_features: pd.DataFrame,
    item_features: pd.DataFrame,
    restaurant_features: pd.DataFrame
) -> List[int]:
    """
    Provides fallback candidates when the standard collaborative/embedding pipeline 
    lacks enough data (e.g., new user, new restaurant).
    Must execute in <40ms.
    """
    candidates = []
    
    is_new_user = user_id not in user_features.index if user_features is not None and not user_features.empty else True
    is_new_rest = restaurant_id not in restaurant_features.index if restaurant_features is not None and not restaurant_features.empty else True
    
    # 1. New Restaurant Fallback: Use category/cuisine popularity mapping (simulated here)
    if is_new_rest and not item_features.empty:
        # Just grab the globally most popular items
        if "popularity_score" in item_features.columns:
            top_global = item_features.sort_values(by="popularity_score", ascending=False).index[:30].tolist()
            candidates.extend(top_global)
            
    # 2. New User Fallback: Suggest very popular safe items within the restaurant
    elif is_new_user and not item_features.empty:
        # We don't have City-level mapping in this simple setup easily accessible here, 
        # so we rely on restaurant-level trending items
        top_rest_items = item_features.sort_values(by="popularity_score", ascending=False).index[:20].tolist()
        candidates.extend(top_rest_items)

    # 3. New Item in Cart Fallback (Cart relies mostly on co-occurrence, which might be 0)
    # If standard retrieval yields nothing, we fallback to category-based similarity
    if not candidates and not item_features.empty:
        # If cart has items, find similar items by category
        if cart_items:
            cart_categories = item_features.loc[cart_items[0], "category"] if cart_items[0] in item_features.index else None
            if cart_categories:
                # Find other popular items in same category
                similar_cat_items = item_features[item_features["category"] == cart_categories]
                if not similar_cat_items.empty:
                    top_sim = similar_cat_items.sort_values(by="popularity_score", ascending=False).index[:20].tolist()
                    candidates.extend(top_sim)

        # Ultimate fallback: Top popular items overall
        if not candidates:
             candidates.extend(item_features.sort_values(by="popularity_score", ascending=False).index[:20].tolist())
             
    # Deduplicate and remove cart items
    candidates = list(set(candidates) - set(cart_items))
    return candidates[:50]  # Return max 50 candidates
