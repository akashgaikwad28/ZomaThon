import pandas as pd
import numpy as np
import uuid
import random
from datetime import datetime, timedelta
import os

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

# Configuration
NUM_USERS = 1000
NUM_RESTAURANTS = 50
NUM_ITEMS_PER_RESTAURANT = (5, 15)
NUM_ORDERS = 5000
OUTPUT_DIR = "data/raw"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_users():
    print(f"Generating {NUM_USERS} users...")
    users = pd.DataFrame({
        "user_id": range(1, NUM_USERS + 1),
        "city": np.random.choice(["Mumbai", "Bangalore", "Delhi", "Hyderabad", "Chennai"], NUM_USERS),
        "is_veg": np.random.choice([0, 1], NUM_USERS, p=[0.7, 0.3]),
        "avg_order_value": np.random.gamma(shape=5, scale=100, size=NUM_USERS)  # Mean ~500
    })
    users.to_parquet(f"{OUTPUT_DIR}/users.parquet")
    return users

def generate_restaurants():
    print(f"Generating {NUM_RESTAURANTS} restaurants...")
    cuisines = ["North Indian", "South Indian", "Chinese", "Fast Food", "Italian", "Desserts", "Biryani"]
    restaurants = pd.DataFrame({
        "restaurant_id": range(1, NUM_RESTAURANTS + 1),
        "cuisine": np.random.choice(cuisines, NUM_RESTAURANTS),
        "rating": np.random.uniform(3.5, 5.0, NUM_RESTAURANTS),
        "city": np.random.choice(["Mumbai", "Bangalore", "Delhi", "Hyderabad", "Chennai"], NUM_RESTAURANTS)
    })
    restaurants.to_parquet(f"{OUTPUT_DIR}/restaurants.parquet")
    return restaurants

def generate_items(restaurants):
    print("Generating items...")
    items_list = []
    
    addon_patterns = {
        "Biryani": ["Salan", "Raita", "Coke", "Thumbs Up", "Gulab Jamun"],
        "Fast Food": ["Fries", "Coke", "Pepsi", "Dips", "Nuggets"],
        "Italian": ["Garlic Bread", "Coke", "Cheese Dip", "Truffle Fries"],
        "Chinese": ["Spring Rolls", "Coke", "Kimchi", "Wontons"],
        "North Indian": ["Naan", "Raita", "Lassi", "Papad"],
        "South Indian": ["Vada", "Coffee", "Buttermilk"],
        "Desserts": ["Extra Scoop", "Chocolate Sauce"]
    }

    for _, res in restaurants.iterrows():
        n_items = random.randint(*NUM_ITEMS_PER_RESTAURANT)
        res_cuisine = res["cuisine"]
        
        # Main items
        for i in range(n_items):
            item_id = len(items_list) + 1
            is_addon = random.random() < 0.2  # 20% items are natural addons
            items_list.append({
                "item_id": item_id,
                "restaurant_id": res["restaurant_id"],
                "item_name": f"{res_cuisine} Item {i}",
                "category": res_cuisine,
                "price": np.random.uniform(100, 600) if not is_addon else np.random.uniform(30, 150),
                "is_veg": random.choice([0, 1]),
                "is_addon": is_addon
            })
            
    items_df = pd.DataFrame(items_list)
    items_df.to_parquet(f"{OUTPUT_DIR}/items.parquet")
    return items_df

def generate_orders(users, restaurants, items):
    print(f"Generating {NUM_ORDERS} orders...")
    orders = []
    
    # Group items by restaurant for fast lookup
    res_items = items.groupby("restaurant_id")["item_id"].apply(list).to_dict()
    item_prices = items.set_index("item_id")["price"].to_dict()
    item_addons = items[items["is_addon"] == True].groupby("restaurant_id")["item_id"].apply(list).to_dict()
    
    start_date = datetime(2026, 1, 1)
    
    for i in range(NUM_ORDERS):
        user_id = random.randint(1, NUM_USERS)
        # Pick a restaurant in the same city (simplified simulation)
        user_city = users.iloc[user_id-1]["city"]
        possible_restaurants = restaurants[restaurants["city"] == user_city]["restaurant_id"].tolist()
        if not possible_restaurants:
            res_id = random.randint(1, NUM_RESTAURANTS)
        else:
            res_id = random.choice(possible_restaurants)
            
        # Time simulation: lunch/dinner peaks
        hour = np.random.choice(
            [12, 13, 14, 19, 20, 21, 22, 15, 16], 
            p=[0.2, 0.2, 0.1, 0.15, 0.15, 0.1, 0.05, 0.03, 0.02]
        )
        order_time = start_date + timedelta(days=random.randint(0, 50), hours=int(hour), minutes=random.randint(0, 59))
        
        # Order items
        available_items = res_items.get(res_id, [])
        if not available_items: continue
        
        # Pick 1-3 main items
        num_main = random.randint(1, 3)
        order_items = random.sample(available_items, min(num_main, len(available_items)))
        
        # Simulation: High probability of add-on if main items are present
        res_addons = item_addons.get(res_id, [])
        if res_addons:
            if random.random() < 0.4: # 40% chance of adding an add-on
                order_items.append(random.choice(res_addons))
        
        total_value = sum(item_prices[it] for it in order_items)
        
        orders.append({
            "order_id": i + 1,
            "user_id": user_id,
            "restaurant_id": res_id,
            "timestamp": order_time,
            "total_value": total_value,
            "items": order_items
        })
        
    orders_df = pd.DataFrame(orders)
    # Explode items for training-ready format
    orders_exploded = orders_df.explode("items").rename(columns={"items": "item_id"})
    
    orders_df.to_parquet(f"{OUTPUT_DIR}/orders.parquet")
    orders_exploded.to_parquet(f"{OUTPUT_DIR}/orders_exploded.parquet")
    print("Done.")

if __name__ == "__main__":
    u = generate_users()
    r = generate_restaurants()
    i = generate_items(r)
    generate_orders(u, r, i)
