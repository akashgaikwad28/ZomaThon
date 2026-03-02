import pandas as pd
import numpy as np
import pickle
import os

# Configuration
RAW_DIR = "data/raw"
MODEL_DIR = "app/models/weights"

os.makedirs(MODEL_DIR, exist_ok=True)

def compute_co_occurrence():
    """
    Computes an item-item co-occurrence matrix based on order sequences.
    This serves as the primary candidate retrieval mechanism.
    """
    print("Loading order data for co-occurrence computation...")
    orders_exploded = pd.read_parquet(f"{RAW_DIR}/orders_exploded.parquet")
    
    # Group items by order_id
    order_groups = orders_exploded.groupby("order_id")["item_id"].apply(list).tolist()
    
    co_occurrence = {}
    
    print(f"Processing {len(order_groups)} orders...")
    for items in order_groups:
        for i in range(len(items)):
            for j in range(len(items)):
                if i == j: continue
                item_a, item_b = items[i], items[j]
                
                if item_a not in co_occurrence:
                    co_occurrence[item_a] = {}
                
                co_occurrence[item_a][item_b] = co_occurrence[item_a].get(item_b, 0) + 1
                
    # Normalize or filter (keep top 10 for each item)
    final_matrix = {}
    for item_id, neighbors in co_occurrence.items():
        sorted_neighbors = sorted(neighbors.items(), key=lambda x: x[1], reverse=True)[:10]
        final_matrix[item_id] = [neighbor_id for neighbor_id, count in sorted_neighbors]
        
    # Save the matrix
    with open(f"{MODEL_DIR}/co_occurrence.pkl", "wb") as f:
        pickle.dump(final_matrix, f)
        
    print(f"Co-occurrence matrix saved to {MODEL_DIR}/co_occurrence.pkl")
    return final_matrix

if __name__ == "__main__":
    compute_co_occurrence()
