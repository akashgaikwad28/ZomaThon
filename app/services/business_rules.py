from typing import List, Dict, Any

class BusinessReRanker:
    """
    Final layer of the recommendation pipeline.
    Applies business constraints like diversity, price filtering, and margin safety.
    """
    
    def __init__(self, diversity_threshold: float = 0.5):
        self.diversity_threshold = diversity_threshold

    def __init__(self, diversity_threshold: float = 0.5):
        self.diversity_threshold = diversity_threshold
        # Same category max limit
        self.max_per_category = 2

    def re_rank(self, recommendations: List[Dict[str, Any]], cart_items: List[int], cart_stage: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Adjusts the order of candidates based on business rules and meal progression.
        """
        from app.features.cart_stage import SIDE_CATEGORIES, DESSERT_CATEGORIES, MAIN_CATEGORIES
        
        # 1. Filter out items already in cart
        filtered_recs = [r for r in recommendations if r["item_id"] not in cart_items]
        
        # 2. Apply Meal Progression Logic (Phase 6)
        if cart_stage:
            for rec in filtered_recs:
                cat = rec.get("category", "")
                
                # Downweight sides if we already have a side
                if cart_stage.get("has_side") and cat in SIDE_CATEGORIES:
                    rec["score"] *= 0.5  # Penalize
                
                # Boost desserts if meal is at least 50% complete and no dessert
                if not cart_stage.get("has_dessert") and cart_stage.get("meal_completion_score", 0) >= 0.5:
                    if cat in DESSERT_CATEGORIES:
                        rec["score"] *= 1.5  # Boost
                        
                # Minor penalty for recommending mains if we already have a main
                if cart_stage.get("has_main") and cat in MAIN_CATEGORIES:
                    rec["score"] *= 0.8

        # 3. Sort by modified model score
        sorted_recs = sorted(filtered_recs, key=lambda x: x["score"], reverse=True)
        
        # 4. Diversity enforcement (Max 2 per category, aim for min 3 unique categories in top 8)
        category_counts = {}
        diverse_recs = []
        
        for rec in sorted_recs:
            category = rec.get("category", "Unknown")
            
            # Skip if we already have max items for this category
            if category_counts.get(category, 0) >= self.max_per_category:
                continue
                
            diverse_recs.append(rec)
            category_counts[category] = category_counts.get(category, 0) + 1
            
            if len(diverse_recs) == 8:
                break
                
        # Fill rest if we filtered too much and fell below 8
        if len(diverse_recs) < 8:
            used_ids = {r["item_id"] for r in diverse_recs}
            for rec in sorted_recs:
                if rec["item_id"] not in used_ids:
                    diverse_recs.append(rec)
                if len(diverse_recs) == 8:
                    break
        
        return diverse_recs

def compute_intra_list_diversity(recommendations: List[Dict[str, Any]]) -> float:
    """
    Computes intra-list diversity based on category uniqueness.
    Returns ratio of unique categories to total items recommended.
    """
    if not recommendations: return 0.0
    unique_categories = len(set([r.get("category", "") for r in recommendations]))
    return float(unique_categories) / len(recommendations)

