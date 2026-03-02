import pandas as pd
import numpy as np
import os
import datetime

# Fixed seed for reproducibility
np.random.seed(42)

# Configuration
NUM_USERS = 10_000
NUM_RESTAURANTS = 500
NUM_ITEMS = 5_000
NUM_ORDERS = 10_000

PROCESSED_DIR = "data/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

# ---------------------------------------------------------
# 1. ENTITY GENERATION
# ---------------------------------------------------------
def generate_users():
    print("Generating users...")
    segments = ['budget', 'premium', 'frequent', 'occasional']
    segment_probs = [0.40, 0.20, 0.25, 0.15]
    
    cities = ['Mumbai', 'Bangalore', 'Pune', 'Delhi']
    
    users = pd.DataFrame({
        'user_id': np.arange(1, NUM_USERS + 1),
        'segment': np.random.choice(segments, NUM_USERS, p=segment_probs),
        'city': np.random.choice(cities, NUM_USERS),
        'is_veg': np.random.choice([0, 1], NUM_USERS, p=[0.6, 0.4])
    })
    
    # Base AOV per user depending on segment
    segment_aov = {'budget': 200, 'premium': 800, 'frequent': 400, 'occasional': 500}
    users['avg_order_value'] = users['segment'].map(segment_aov) * np.random.normal(1.0, 0.2, NUM_USERS)
    users['avg_order_value'] = users['avg_order_value'].clip(lower=100)
    
    return users

def generate_restaurants():
    print("Generating restaurants...")
    cities = ['Mumbai', 'Bangalore', 'Pune', 'Delhi']
    
    restaurants = pd.DataFrame({
        'restaurant_id': np.arange(1, NUM_RESTAURANTS + 1),
        'city': np.random.choice(cities, NUM_RESTAURANTS),
        'base_rating': np.random.uniform(3.0, 5.0, NUM_RESTAURANTS).round(1),
        'is_new': np.random.choice([True, False], NUM_RESTAURANTS, p=[0.05, 0.95]) # 5% new for cold start
    })
    return restaurants

def generate_items(restaurants):
    print("Generating items...")
    categories = ['Main Course', 'Starters', 'Desserts', 'Beverages', 'Breads', 'Biryani', 'Pizza', 'Sides']
    cat_weights = [0.3, 0.2, 0.1, 0.15, 0.1, 0.1, 0.03, 0.02]
    
    items = pd.DataFrame({
        'item_id': np.arange(1, NUM_ITEMS + 1),
        'restaurant_id': np.random.choice(restaurants['restaurant_id'], NUM_ITEMS),
        'category': np.random.choice(categories, NUM_ITEMS, p=cat_weights),
        'is_veg': np.random.choice([0, 1], NUM_ITEMS, p=[0.5, 0.5]),
        'is_new': np.random.choice([True, False], NUM_ITEMS, p=[0.05, 0.95]) # 5% unseen items
    })
    
    # Pricing logic based on category
    cat_base_price = {
        'Main Course': 250, 'Starters': 180, 'Desserts': 150, 
        'Beverages': 80, 'Breads': 40, 'Biryani': 300, 
        'Pizza': 400, 'Sides': 100
    }
    items['price'] = items['category'].map(cat_base_price) * np.random.normal(1.0, 0.3, NUM_ITEMS)
    items['price'] = items['price'].clip(lower=20).round(0)
    
    return items

# ---------------------------------------------------------
# 2. ORDER LOGIC & MEAL PROGRESSION
# ---------------------------------------------------------
def generate_orders(users, restaurants, items):
    print("Generating orders (vectorized)...")
    
    # 3. CONTEXT (Time & City distributions)
    active_users = users['user_id'].values[:int(NUM_USERS * 0.90)]
    order_user_ids = np.random.choice(active_users, NUM_ORDERS)
    
    base_times = pd.date_range(end=pd.Timestamp.now(), periods=NUM_ORDERS, freq='s')
    
    # Pre-select restaurants for orders
    order_rest_ids = np.random.choice(restaurants['restaurant_id'].values[10:], NUM_ORDERS)
    
    orders = pd.DataFrame({
        'order_id': np.arange(1, NUM_ORDERS + 1),
        'user_id': order_user_ids,
        'restaurant_id': order_rest_ids,
        'timestamp': base_times
    })
    
    orders = orders.merge(users[['user_id', 'city', 'segment', 'avg_order_value']], on='user_id', how='left')
    
    # To vectorize Item Selection, we create pre-mapped arrays
    main_items = items[items['category'].isin(['Main Course', 'Biryani', 'Pizza'])]['item_id'].values
    side_items = items[items['category'].isin(['Starters', 'Sides', 'Breads'])]['item_id'].values
    bev_items = items[items['category'].isin(['Beverages'])]['item_id'].values
    des_items = items[items['category'].isin(['Desserts'])]['item_id'].values
    
    # Base item logic probabilities
    prob_main = 0.6
    prob_side_if_main = 0.7
    prob_bev = 0.5
    prob_des = 0.3
    
    # Generate vectors of booleans for 1M rows
    rands = np.random.random(size=(NUM_ORDERS, 4))
    
    has_main = rands[:, 0] < prob_main
    has_side = (rands[:, 1] < prob_side_if_main) & has_main
    
    des_modifier = np.zeros(NUM_ORDERS)
    des_modifier[orders['segment'] == 'premium'] += 0.075 # +25% relative (approx)
    des_modifier[has_side] += 0.20 # +20% absolute
    has_des = rands[:, 2] < (prob_des + des_modifier)
    has_bev = rands[:, 3] < prob_bev
    
    # Fast assign random items from global pool based on flags
    # Real-world assigns by rest_id, but globally selecting here runs 5000x faster for synthetic rules
    import gc
    gc.collect()
    
    def get_items(flag_array, choices):
        num_true = flag_array.sum()
        res = np.zeros(NUM_ORDERS, dtype=np.int64)
        res[flag_array] = np.random.choice(choices, num_true)
        return res
        
    main_choices = get_items(has_main, main_items)
    side_choices = get_items(has_side, side_items)
    des_choices = get_items(has_des, des_items)
    bev_choices = get_items(has_bev, bev_items)
    
    # Expand to rows
    orders['main'] = main_choices
    orders['side'] = side_choices
    orders['des'] = des_choices
    orders['bev'] = bev_choices
    
    # Price rejection modeling (vectorized)
    item_prices = items.set_index('item_id')['price']
    
    # Re-map prices back to vectors to check
    # If budget and price > 1.5x AOV, set to 0
    budget_mask = orders['segment'] == 'budget'
    thresh = orders['avg_order_value'] * 1.5
    
    for col in ['main', 'side', 'des', 'bev']:
        prices = orders[col].map(item_prices).fillna(0)
        reject = budget_mask & (prices > thresh)
        orders.loc[reject, col] = 0
        
    # Ensure at least 1 item per order (if all 0, give random main)
    empty_mask = (orders['main'] == 0) & (orders['side'] == 0) & (orders['des'] == 0) & (orders['bev'] == 0)
    orders.loc[empty_mask, 'main'] = np.random.choice(main_items, empty_mask.sum())
    
    # Melt dataframe to create exploded view
    id_vars = ['order_id', 'user_id', 'restaurant_id', 'timestamp']
    exploded = pd.melt(orders, id_vars=id_vars, value_vars=['main', 'side', 'des', 'bev'], value_name='item_id')
    exploded = exploded[exploded['item_id'] != 0].copy()
    exploded['sequence_step'] = exploded.groupby('order_id').cumcount() + 1
    exploded = exploded.sort_values(by=['order_id', 'sequence_step'])
    
    return exploded

if __name__ == "__main__":
    users = generate_users()
    restaurants = generate_restaurants()
    items = generate_items(restaurants)
    
    users.to_parquet(f"{PROCESSED_DIR}/users_adv.parquet")
    restaurants.to_parquet(f"{PROCESSED_DIR}/restaurants_adv.parquet")
    items.to_parquet(f"{PROCESSED_DIR}/items_adv.parquet")
    
    orders = generate_orders(users, restaurants, items)
    orders.to_parquet(f"{PROCESSED_DIR}/orders_exploded_adv.parquet")
    
    print("Advanced Data Generation Complete.")
