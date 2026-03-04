# ZomaThon: End-to-End Workflow (Deep Dive)

This document provides a rigorous technical breakdown of the four distinct operational phases powering the Cart Super Add-On (CSAO) engine. Because no proprietary Zomato data was available, the entire workflow—from mathematically constrained synthetic generation to offline GMV lifting—was engineered from scratch.

## End-to-End Workflow Diagram

<iframe src="assets/workflowdiagram.html" width="100%" height="800px" style="border:none;"></iframe>

---

## Phase 1: Offline Data Generation & Preprocessing

The foundation of the recommendation pipeline requires simulating a "realistic" food delivery environment. Randomly distributing item additions creates meaningless noise that $O(N)$ ML models cannot learn from. 

We engineered `advanced_data_generator.py` to enforce strict real-world economic and behavioral constraints across 10,000 synthetic orders:

### 1. The Markov-Chain "Incomplete Meal" Simulator
*   **Logic**: If a user adds a `Main Course` to their cart, the algorithm uses transition probabilities to determine the next item. Crucially, the code intentionally terminates the progression prematurely exactly 50% of the time. 
*   **Why?**: This manufactures historically "incomplete" carts (e.g., a Biryani without a beverage). This serves as the exact *negative sample* signal the LightGBM model needs to learn cart-completion logic.

### 2. Time Bimodality & Price Rejectors
*   **Temporal Spikes**: Orders are generated following a bimodal Gumbel distribution, severely clustering traffic around Lunch (13:00) and Dinner (20:30).
*   **Price Elasticity**: The generator assigns users into 4 economic tiers (Budget to Premium). If a user is "Budget" and the RNG attempts to add a ₹400 dessert to a ₹200 cart, the logic explicitly rejects the addition via vector masking, simulating real-world price shock.

### Preprocessing & Feature Engineering
`preprocess.py` processes the raw Parquet lakes into structured feature matrices:
*   Aggregating historical AOV and total order counts per user.
*   Pre-computing static similarity scores (e.g., matching a vegetarian user against a vegetarian menu item).

---

## Phase 2: Offline Model Training

The `processed/` feature lake feeds three heavily decoupled training scripts. 

### 1. Semantic Embedding Extraction (`train_item2vec.py`)
Applying Word2Vec (Skip-Gram) logic to E-Commerce. We treat user carts as "sentences." This trains a 64-dimensional latent embedding vector for every item. Even if a niche item isn't bought often, if it's bought in similar *contexts* to popular items, it sits close to them in the vector space, solving the Long Tail recommendation problem.

### 2. Co-Occurrence Compiling (`co_occurrence.py`)
A fast, deterministic adjacency list mapping `$Item_A \rightarrow [Item_B, Item_C]$` based purely on historical pair frequencies. This array is exported to `.pkl` for subsequent Redis ingestion.

### 3. LightGBM Learning (`train_lgbm.py`)
The 22-dimensional feature matrix is fed into a Microsoft LightGBM algorithm. We explicitly configure the model with the `lambdarank` objective function, punishing the model mathematically if it predicts highly relevant additive items outside the visible UI boundary (ranks $1-8$).

---

## Phase 3: Online Real-Time Inference (FastAPI)

When a User queries the `/api/v1/recommend` endpoint, the system must execute the Full-Stack funnel in **<250ms SLA**.

1.  **Context Construction**: The API instantly retrieves the User's AOV and Veg Preferences from the PostgreSQL/Parquet Feature Store into working RAM.
2.  **O(1) Retrieval**: The candidate pool is reduced from $5000 \rightarrow 50$ via sub-5ms Redis cache lookups across the Co-Occurrence arrays.
3.  **LightGBM Execution**: The 22-dimensional matrix drops through the compiled decision trees ($<20$ms CPU time) yielding purely statistical Propensity Logits.
4.  **Business Filter**: The $O(N \log N)$ Array Re-Ranker activates, enforcing UX diversity (Max 2 breads), blocking redundancy (No extra sides), and mathematically multiplying high-margin dessert scores if the cart is complete.
5.  **Return**: The Top 8 candidates are returned as JSON, and a detached BackgroundThread fires the prompt to Gemini/Groq for explanation generation.

---

## Phase 4: Offline Evaluation & Business Simulation

Before any A/B testing can be approved, the system proves its worth offline against a strict chronologically-split holdout set.

### 1. Removing Classification Bias
The evaluation framework explicitly ignores standard ML accuracy metrics like AUC. A model that predicts a highly relevant item at position #25 looks great on AUC, but is invisible to the user on the UI rail. 
We strictly track **NDCG@8** (Normalized Discounted Cumulative Gain) and **Precision@8** against a native Localized Popularity Baseline.

### 2. Simulating GMV (AOV-Lift)
*   **The Problem**: Data Scientists care about Precision. Stakeholders care about Gross Merchandise Value (GMV).
*   **The Solution**: For every True Positive (a recommended item the user *actually bought* historically), the script dynamically sums the explicit Rupee (`price`) value. This models exactly how much extra pure revenue the CSAO engine will drive per-order upon deployment.
