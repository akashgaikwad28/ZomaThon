import pandas as pd
import json
import os

PROCESSED_DIR = "data/processed"
RESULTS_DIR = "experiments"

os.makedirs(RESULTS_DIR, exist_ok=True)

def run_validation():
    print("Loading data for validation...")
    try:
        users = pd.read_parquet(f"{PROCESSED_DIR}/users_adv.parquet")
        items = pd.read_parquet(f"{PROCESSED_DIR}/items_adv.parquet")
        orders = pd.read_parquet(f"{PROCESSED_DIR}/orders_exploded_adv.parquet")
    except FileNotFoundError:
        print("Data not found. Run advanced_data_generator.py first.")
        return

    print("Analyzing...")
    report = {}
    
    # 1. User Segment Balance
    segment_counts = users['segment'].value_counts(normalize=True).to_dict()
    report["user_segment_balance"] = {k: round(v, 4) for k, v in segment_counts.items()}
    
    # 2. Veg vs Non-Veg Ratio
    veg_counts = items['is_veg'].value_counts(normalize=True).to_dict()
    report["item_veg_ratio"] = {"Vegetarian": round(veg_counts.get(1, 0), 4), "Non-Vegetarian": round(veg_counts.get(0, 0), 4)}
    
    # 3. Order Size Distribution
    order_sizes = orders.groupby('order_id').size()
    report["order_size_distribution"] = {
        "mean_items_per_order": round(order_sizes.mean(), 2),
        "max_items_per_order": int(order_sizes.max()),
        "single_item_orders_pct": round((order_sizes == 1).sum() / len(order_sizes), 4)
    }
    
    # 4. Category Balance
    orders_with_cat = orders.merge(items[['item_id', 'category']], on='item_id')
    cat_counts = orders_with_cat['category'].value_counts(normalize=True).to_dict()
    report["cart_category_distribution"] = {k: round(v, 4) for k, v in cat_counts.items()}
    
    # 5. Peak Hour Traffic Spikes
    orders['hour'] = orders['timestamp'].dt.hour
    hour_counts = orders['hour'].value_counts(normalize=True).sort_index()
    # Find peak hours (top 3)
    peak_hours = hour_counts.nlargest(3).index.tolist()
    report["traffic_spikes"] = {
        "peak_hours_24h": [int(h) for h in peak_hours],
        "is_lunch_peak_present": bool(any(12 <= h <= 15 for h in peak_hours)),
        "is_dinner_peak_present": bool(any(19 <= h <= 22 for h in peak_hours))
    }
    
    # 6. Sequential Integrity (Main course usually before side/dessert)
    # Check if a Main Course appears at sequence_step 1 or 2 more often than > 2
    main_orders = orders_with_cat[orders_with_cat['category'].isin(['Main Course', 'Biryani', 'Pizza'])]
    if not main_orders.empty:
        early_mains = (main_orders['sequence_step'] <= 2).sum()
        total_mains = len(main_orders)
        report["sequential_integrity"] = {
            "mains_ordered_early_pct": float(round(early_mains / total_mains, 4)),
            "passed_sanity_check": bool((early_mains / total_mains) > 0.8) # Expecting > 80%
        }
    
    # Save Report
    output_path = f"{RESULTS_DIR}/data_quality_report.json"
    with open(output_path, "w") as f:
        json.dump(report, f, indent=4)
        
    print(f"Validation complete. Report saved to {output_path}")
    print(json.dumps(report, indent=4))

if __name__ == "__main__":
    run_validation()
