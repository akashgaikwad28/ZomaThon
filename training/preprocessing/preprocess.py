import pandas as pd
import numpy as np
import os

# Configuration
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

os.makedirs(PROCESSED_DIR, exist_ok=True)

def load_data():
    users = pd.read_parquet(f"{PROCESSED_DIR}/users_adv.parquet")
    restaurants = pd.read_parquet(f"{PROCESSED_DIR}/restaurants_adv.parquet")
    items = pd.read_parquet(f"{PROCESSED_DIR}/items_adv.parquet")
    orders_exploded = pd.read_parquet(f"{PROCESSED_DIR}/orders_exploded_adv.parquet")
    return users, restaurants, items, orders_exploded

def compute_restaurant_features(restaurants, orders_exploded):
    print("Computing restaurant features...")
    # Restaurant popularity
    rest_pop = orders_exploded.groupby("restaurant_id")["order_id"].count().reset_index()
    rest_pop.rename(columns={"order_id": "restaurant_popularity_score"}, inplace=True)
    
    rest_features = restaurants.merge(rest_pop, on="restaurant_id", how="left").fillna(0)
    rest_features.to_parquet(f"{PROCESSED_DIR}/restaurant_features.parquet")
    return rest_features

def compute_user_features(users, orders_exploded, items):
    print("Computing user features...")
    
    # Base frequency
    user_freq = orders_exploded.groupby("user_id")["order_id"].nunique().reset_index()
    user_freq.rename(columns={"order_id": "order_count"}, inplace=True)
    
    # Recency & Frequency/Month
    user_last_order = orders_exploded.groupby("user_id")["timestamp"].max().reset_index()
    user_last_order.rename(columns={"timestamp": "last_order_ts"}, inplace=True)
    
    user_features = users.merge(user_freq, on="user_id", how="left").fillna(0)
    user_features = user_features.merge(user_last_order, on="user_id", how="left")
    
    current_time = pd.Timestamp.now()
    user_features["recency_days"] = (current_time - user_features["last_order_ts"]).dt.days.fillna(999)
    user_features["frequency_per_month"] = user_features["order_count"] / 1.0 # Assuming 1 month of data
    
    # Category add rates (Desserts / Beverages)
    orders_with_cat = orders_exploded.merge(items[["item_id", "category"]], on="item_id")
    
    # Count total items per user
    user_total_items = orders_exploded.groupby("user_id")["item_id"].count().reset_index(name="total_items_bought")
    
    # Count dessert/bev items per user
    user_desserts = orders_with_cat[orders_with_cat["category"] == "Desserts"].groupby("user_id")["item_id"].count().reset_index(name="dessert_count")
    user_bevs = orders_with_cat[orders_with_cat["category"] == "Beverages"].groupby("user_id")["item_id"].count().reset_index(name="bev_count")
    
    user_features = user_features.merge(user_total_items, on="user_id", how="left").fillna(0)
    user_features = user_features.merge(user_desserts, on="user_id", how="left").fillna(0)
    user_features = user_features.merge(user_bevs, on="user_id", how="left").fillna(0)
    
    user_features["dessert_add_rate"] = np.where(user_features["total_items_bought"] > 0, user_features["dessert_count"] / user_features["total_items_bought"], 0.0)
    user_features["beverage_add_rate"] = np.where(user_features["total_items_bought"] > 0, user_features["bev_count"] / user_features["total_items_bought"], 0.0)
    
    # Price Elasticity Score
    user_features["price_elasticity_score"] = np.random.uniform(0.5, 2.0, len(user_features)) # In real life, compute std_dev of item prices / mean
    user_features["is_veg_user"] = (np.random.rand(len(user_features)) > 0.6).astype(int)
    user_features["avg_order_value"] = np.random.normal(300, 50, len(user_features))
    
    user_features.to_parquet(f"{PROCESSED_DIR}/user_features.parquet")
    return user_features

def compute_item_features(items, orders_exploded):
    print("Computing item features...")
    # Item popularity
    item_pop = orders_exploded.groupby("item_id")["order_id"].count().reset_index()
    item_pop.rename(columns={"order_id": "popularity_score"}, inplace=True)
    
    item_features = items.merge(item_pop, on="item_id", how="left").fillna(0)
    
    # New Item Context
    item_features["popularity_percentile"] = item_features["popularity_score"].rank(pct=True)
    item_features["seasonal_score"] = np.random.uniform(0.8, 1.2, len(item_features)) # Mock seasonal multiplier
    item_features["margin_estimate"] = item_features["price"] * np.random.uniform(0.15, 0.40, len(item_features)) # 15-40% margin
    
    item_features.rename(columns={"is_veg": "is_veg_item"}, inplace=True)
    item_features.to_parquet(f"{PROCESSED_DIR}/item_features.parquet")
    return item_features

def create_training_dataset(orders_exploded, user_features, item_features, rest_features):
    print("Creating training dataset (positives and negatives)...")
    
    # Positive samples (actual orders)
    positives = orders_exploded[["user_id", "item_id", "timestamp"]].copy()
    positives["label"] = 1
    
    # Negative sampling (simplified for hackathon)
    # For each positive, pick 2 random items the user NOT ordered
    all_item_ids = item_features["item_id"].unique()
    
    negatives = []
    for _, row in positives.iloc[:10000].iterrows(): # sampling first 10k for speed
        neg_items = np.random.choice(all_item_ids, 2, replace=False)
        for neg_item in neg_items:
            negatives.append({
                "user_id": row["user_id"],
                "item_id": neg_item,
                "timestamp": row["timestamp"],
                "label": 0
            })
            
    neg_df = pd.DataFrame(negatives)
    train_df = pd.concat([positives.iloc[:10000], neg_df]).sample(frac=1).reset_index(drop=True)
    
    # Merge features
    train_df = train_df.merge(user_features, on="user_id", how="left")
    train_df = train_df.merge(item_features, on="item_id", how="left")
    train_df = train_df.merge(rest_features[["restaurant_id", "restaurant_popularity_score"]], on="restaurant_id", how="left").fillna(0)
    
    # Advanced Personalization Features (Phase 6)
    train_df["hour_bucket"] = train_df["timestamp"].dt.hour // 4  # 6 buckets per day
    train_df["weekend_flag"] = (train_df["timestamp"].dt.dayofweek >= 5).astype(int)
    
    # Simple meal type (1=Lunch, 2=Dinner, 0=Other)
    train_df["meal_type"] = np.where((train_df["timestamp"].dt.hour >= 12) & (train_df["timestamp"].dt.hour <= 15), 1, 
                            np.where((train_df["timestamp"].dt.hour >= 19) & (train_df["timestamp"].dt.hour <= 22), 2, 0))
                            
    train_df["price_deviation_from_user_avg"] = train_df["price"] - train_df["avg_order_value"]
    train_df["veg_alignment_score"] = (train_df["is_veg_user"] == train_df["is_veg_item"]).astype(int)
    
    # Mock city embedding (scalar for simplicity)
    city_map = {'Mumbai': 1.0, 'Bangalore': 2.0, 'Pune': 3.0, 'Delhi': 4.0}
    train_df["city_embedding"] = train_df["city"].map(city_map).fillna(0.0)
    
    # Mock user_item_interaction_count for speed in hackathon (random or based on pop)
    train_df["user_item_interaction_count"] = np.random.randint(0, 5, size=len(train_df))
    
    train_df.to_parquet(f"{PROCESSED_DIR}/train_set.parquet")
    print(f"Training set created with {len(train_df)} rows.")

if __name__ == "__main__":
    try:
        u, r, i, o = load_data()
        u_feat = compute_user_features(u, o, i)
        i_feat = compute_item_features(i, o)
        r_feat = compute_restaurant_features(r, o)
        create_training_dataset(o, u_feat, i_feat, r_feat)
    except FileNotFoundError:
        print("Raw data not found. Please run generate_data.py first.")
