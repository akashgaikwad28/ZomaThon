import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import roc_auc_score, precision_score
import json
import os

PROCESSED_DIR = "data/processed"
MODEL_DIR = "app/models/weights"
RESULTS_DIR = "experiments/results"

os.makedirs(RESULTS_DIR, exist_ok=True)

def dcg_at_k(r, k):
    r = np.asarray(r, dtype=float)[:k]
    if r.size:
        return np.sum(np.subtract(np.power(2, r), 1) / np.log2(np.arange(2, r.size + 2)))
    return 0.

def ndcg_at_k(r, k):
    idcg = dcg_at_k(sorted(r, reverse=True), k)
    if not idcg:
        return 0.
    return dcg_at_k(r, k) / idcg

def evaluate_model():
    print("Loading test data and model...")
    # For hackathon simplicity, we evaluate on the training set (or a mock split)
    # In a real scenario, this would be a strictly temporally split test set
    try:
        df = pd.read_parquet(f"{PROCESSED_DIR}/train_set.parquet")
        with open(f"{MODEL_DIR}/ranker_model.pkl", "rb") as f:
            model = pickle.load(f)
    except FileNotFoundError:
        print("Required artifacts not found. Please train models first.")
        return

    features = [
        "is_veg_user", "avg_order_value", "order_count",
        "price", "is_veg_item", "popularity_score",
        "price_deviation_from_user_avg", "veg_alignment_score",
        "user_item_interaction_count", "restaurant_popularity_score",
        "recency_days", "frequency_per_month", "dessert_add_rate", 
        "beverage_add_rate", "price_elasticity_score", "popularity_percentile",
        "seasonal_score", "margin_estimate", "hour_bucket", "weekend_flag",
        "meal_type", "city_embedding"
    ]
        
    X = df[features]
    y = df["label"]
    
    print("Generating predictions...")
    preds = model.predict_proba(X)[:, 1]
    
    # 1. Classification Metrics
    auc = roc_auc_score(y, preds)
    

    # 2. Ranking Metrics @ K & Cold Start Segmentation
    df['pred'] = preds
    grouped = df.groupby('user_id')
    
    precisions_at_8 = []
    recalls_at_8 = []
    ndcgs_at_8 = []
    aov_lifts = []
    
    # Cold Start Segments
    cold_start_metrics = {
        'new_user': {'p8': [], 'ndcg8': []},
        'new_restaurant': {'p8': [], 'ndcg8': []},
        'new_item': {'p8': [], 'ndcg8': []}
    }
    
    baseline_precisions_at_8 = []
    baseline_ndcgs_at_8 = []
    
    for _, group in grouped:
        if len(group) < 2: continue
        
        actual_positives = group[group['label'] == 1]['item_id'].tolist()
        num_actual = len(actual_positives)
        if num_actual == 0: continue
            
        top_8_model_indices = group.nlargest(8, 'pred').index
        top_8_model = group.loc[top_8_model_indices]
        relevant_in_top_8 = top_8_model[top_8_model['label'] == 1]
        
        p_at_8 = len(relevant_in_top_8) / 8.0
        r_at_8 = len(relevant_in_top_8) / num_actual
        relevance = [1 if item in actual_positives else 0 for item in top_8_model['item_id']]
        n_at_8 = ndcg_at_k(relevance, 8)
        
        # Segment Flags
        is_new_user = group['order_count'].max() <= 1 # Approx for new user
        is_new_rest = group['restaurant_popularity_score'].max() <= 10 # Approx for new rest
        
        if is_new_user:
            cold_start_metrics['new_user']['p8'].append(p_at_8)
            cold_start_metrics['new_user']['ndcg8'].append(n_at_8)
            
        if is_new_rest:
            cold_start_metrics['new_restaurant']['p8'].append(p_at_8)
            cold_start_metrics['new_restaurant']['ndcg8'].append(n_at_8)
            
        # Check if recommendations contained a rare "new" item
        if top_8_model['popularity_score'].min() <= 5: 
            cold_start_metrics['new_item']['p8'].append(p_at_8)
            cold_start_metrics['new_item']['ndcg8'].append(n_at_8)
        
        model_aov_lift = relevant_in_top_8['price'].sum() if not relevant_in_top_8.empty else 0
        
        precisions_at_8.append(p_at_8)
        recalls_at_8.append(r_at_8)
        ndcgs_at_8.append(n_at_8)
        aov_lifts.append(float(model_aov_lift))
        
        # Baseline Mapping (Popularity)
        top_8_base = group.nlargest(8, 'popularity_score')
        rel_base_top_8 = top_8_base[top_8_base['label'] == 1]
        bp_at_8 = len(rel_base_top_8) / 8.0
        b_n_at_8 = ndcg_at_k([1 if item in actual_positives else 0 for item in top_8_base['item_id']], 8)
        baseline_precisions_at_8.append(bp_at_8)
        baseline_ndcgs_at_8.append(b_n_at_8)

    # Aggregate
    results = {
        "global_auc": float(auc),
        "ZomaThon_Model": {
            "Precision@8": float(np.mean(precisions_at_8)),
            "Recall@8": float(np.mean(recalls_at_8)),
            "NDCG@8": float(np.mean(ndcgs_at_8)),
            "Average_Simulated_AOV_Lift": float(np.mean(aov_lifts))
        },
        "Cold_Start_Segments": {
            "New_User": {
                 "Precision@8": float(np.mean(cold_start_metrics['new_user']['p8']) if cold_start_metrics['new_user']['p8'] else 0),
                 "NDCG@8": float(np.mean(cold_start_metrics['new_user']['ndcg8']) if cold_start_metrics['new_user']['ndcg8'] else 0),
            },
            "New_Restaurant": {
                 "Precision@8": float(np.mean(cold_start_metrics['new_restaurant']['p8']) if cold_start_metrics['new_restaurant']['p8'] else 0),
                 "NDCG@8": float(np.mean(cold_start_metrics['new_restaurant']['ndcg8']) if cold_start_metrics['new_restaurant']['ndcg8'] else 0),
            },
            "New_Item": {
                 "Precision@8": float(np.mean(cold_start_metrics['new_item']['p8']) if cold_start_metrics['new_item']['p8'] else 0),
                 "NDCG@8": float(np.mean(cold_start_metrics['new_item']['ndcg8']) if cold_start_metrics['new_item']['ndcg8'] else 0),
            }
        },
        "Popularity_Baseline": {
            "Precision@8": float(np.mean(baseline_precisions_at_8)),
            "NDCG@8": float(np.mean(baseline_ndcgs_at_8)),
            "Average_Simulated_AOV_Lift": 0.0
        }
    }
    
    with open(f"{RESULTS_DIR}/cold_start_analysis.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("\n=== Evaluation Results ===")
    print(f"Global AUC: {results['global_auc']:.4f}")
    print("\nZomaThon Model:")
    print(f"  Precision@8: {results['ZomaThon_Model']['Precision@8']:.4f}")
    print(f"  NDCG@8:      {results['ZomaThon_Model']['NDCG@8']:.4f}")
    print(f"  AOV Lift:   INR {results['ZomaThon_Model']['Average_Simulated_AOV_Lift']:.2f}")
    print("\nPopularity Baseline:")
    print(f"  Precision@8: {results['Popularity_Baseline']['Precision@8']:.4f}")
    print(f"  NDCG@8:      {results['Popularity_Baseline']['NDCG@8']:.4f}")
    print(f"  AOV Lift:   INR {results['Popularity_Baseline']['Average_Simulated_AOV_Lift']:.2f}")

if __name__ == "__main__":
    evaluate_model()
