import pandas as pd
from typing import List, Dict, Any

# Define category mappings for meal progression
MAIN_CATEGORIES = {"Biryani", "Pizza", "Burger", "Main Course", "Thali", "Rolls", "Wraps"}
SIDE_CATEGORIES = {"Starters", "Snacks", "Sides", "Breads", "Fries", "Salan", "Raita"}
DESSERT_CATEGORIES = {"Desserts", "Ice Cream", "Sweets", "Cakes", "Mithai"}
BEVERAGE_CATEGORIES = {"Beverages", "Drinks", "Shakes", "Cold Coffee"}

def detect_cart_stage(cart_items: List[int], item_metadata: pd.DataFrame) -> Dict[str, Any]:
    """
    Detects the current stage of the meal based on items in the cart.
    Returns flags for presence of main, side, dessert, beverage, and a completion score.
    """
    # If using dicts/API data instead of pandas df, adapt this
    # For now, assume item_metadata is the self.item_features pandas DataFrame
    
    flags = {
        "has_main": False,
        "has_side": False,
        "has_dessert": False,
        "has_beverage": False,
        "meal_completion_score": 0.0
    }
    
    if not cart_items or item_metadata is None or item_metadata.empty:
        return flags

    categories_present = set()
    
    # Iterate through items to find categories
    for item_id in cart_items:
        if item_id in item_metadata.index:
            cat = item_metadata.loc[item_id, "category"]
            categories_present.add(cat)
            
            if cat in MAIN_CATEGORIES:
                flags["has_main"] = True
            elif cat in SIDE_CATEGORIES:
                flags["has_side"] = True
            elif cat in DESSERT_CATEGORIES:
                flags["has_dessert"] = True
            elif cat in BEVERAGE_CATEGORIES:
                flags["has_beverage"] = True
            else:
                # Basic fallback if categories don't match specific sets perfectly
                # Let's say we assume any other big category is a main
                if pd.notna(cat) and "Main" in str(cat):
                    flags["has_main"] = True

    # Calculate meal completion score: (# distinct meta-categories present) / 4.0
    meta_categories_present = sum([
        flags["has_main"],
        flags["has_side"],
        flags["has_dessert"],
        flags["has_beverage"]
    ])
    
    flags["meal_completion_score"] = float(meta_categories_present) / 4.0
    
    return flags
