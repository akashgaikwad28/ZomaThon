import pandas as pd
import os
import pickle
from gensim.models import Word2Vec

PROCESSED_DIR = "data/processed"
MODEL_DIR = "app/models/weights"

os.makedirs(MODEL_DIR, exist_ok=True)

def train_item2vec():
    """
    Trains Word2Vec embeddings on historical order sequences.
    Each user's daily orders form a sequence.
    """
    print("Loading raw order data for Item2Vec...")
    try:
        orders_exploded = pd.read_parquet(f"{PROCESSED_DIR}/orders_exploded_adv.parquet")
    except FileNotFoundError:
        print("Raw data not found. Please run generate_data.py first.")
        return

    print("Building sequences...")
    # Group by user_id and day to form sequences of items
    orders_exploded['date'] = orders_exploded['timestamp'].dt.date
    
    # Sort by timestamp to ensure sequence order logic
    orders_exploded = orders_exploded.sort_values(by=['user_id', 'timestamp'])
    
    # Create list of lists representing orders
    sequences_df = orders_exploded.groupby(['user_id', 'date'])['item_id'].apply(list)
    
    # Convert item_ids to strings for Gensim Word2Vec
    sentences = [[str(item) for item in sequence] for sequence in sequences_df if len(sequence) > 1]
    
    print(f"Training on {len(sentences)} sequences...")
    
    # User requested: window=3, dimension=64
    model = Word2Vec(
        sentences=sentences,
        vector_size=64,
        window=3,
        min_count=1,
        workers=4,
        sg=1 # skip-gram usually better for recommendations
    )
    
    # Extract just the vectors into a dictionary for fast inference
    embeddings_dict = {}
    for item_str in model.wv.index_to_key:
        embeddings_dict[int(item_str)] = model.wv[item_str]
        
    # Save the embeddings dictionary
    output_path = f"{MODEL_DIR}/item_embeddings.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(embeddings_dict, f)
        
    print(f"Successfully trained and saved {len(embeddings_dict)} item embeddings to {output_path}")

if __name__ == "__main__":
    train_item2vec()
