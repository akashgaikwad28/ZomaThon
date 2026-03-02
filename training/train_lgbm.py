import pandas as pd
import lightgbm as lgb
import os
import pickle

# Configuration
PROCESSED_DIR = "data/processed"
MODEL_DIR = "app/models/weights"

os.makedirs(MODEL_DIR, exist_ok=True)

def train_ranker():
    """
    Trains a LightGBM ranking model using the processed training set.
    """
    print("Loading training data...")
    train_df = pd.read_parquet(f"{PROCESSED_DIR}/train_set.parquet")
    
    # Feature selection
    features = [
        "is_veg_user", "avg_order_value", "order_count",
        "price", "is_veg_item", "popularity_score",
        "price_deviation_from_user_avg", "veg_alignment_score",
        "user_item_interaction_count", "restaurant_popularity_score",
        
        # Phase 7
        "recency_days", "frequency_per_month", "dessert_add_rate", 
        "beverage_add_rate", "price_elasticity_score", "popularity_percentile",
        "seasonal_score", "margin_estimate", "hour_bucket", "weekend_flag",
        "meal_type", "city_embedding"
    ]
    target = "label"
    
    print(f"Training on {len(train_df)} samples with {len(features)} features...")
    
    # Convert categorical if needed (here we treat them as numeric for simplicity)
    X = train_df[features]
    y = train_df[target]
    
    # Train model
    model = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        num_leaves=31,
        random_state=42
    )
    
    model.fit(X, y)
    
    # Save model
    with open(f"{MODEL_DIR}/ranker_model.pkl", "wb") as f:
        pickle.dump(model, f)
        
    print(f"Ranking model saved to {MODEL_DIR}/ranker_model.pkl")

if __name__ == "__main__":
    train_ranker()
