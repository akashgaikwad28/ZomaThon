# ZomaThon: ML Model Architecture & Funnel (Deep Dive)

This document provides an exhaustive technical breakdown of the 4-layer funnel architecture used by the ZomaThon Cart Super Add-On (CSAO) engine. The primary architectural philosophy is **Gray-Box Recommendation**, balancing pure Machine Learning representations (Item2Vec, LightGBM) with deterministic, explicitly encoded business constraints.

## The 4-Layer Funnel Diagram

<iframe src="assets/modelarchitecture.html" width="100%" height="800px" style="border:none;"></iframe>

---

## Layer 1: Candidate Generation (Retrieval)
A standard food delivery catalog contains thousands of items. Executing a highly-dimensional tree-based ML model across all $5,000$ items natively triggers an $O(N)$ CPU bottleneck, drastically violating our $<250$ms SLA.

Instead, Layer 1 executes an $O(1)$ memory lookup to filter the $5,000$ possible items down to a dense pool of $\sim 50$ highly relevant candidates. We achieve this through two parallel streams:

### 1. Item-Item Co-Occurrence Graph
*   **Logic**: Parses the historical transaction database to build a normalized transition matrix. If a user has `Chicken Biryani` in their cart, what are the top items historically bought alongside it?
*   **Data Structure**: Pre-calculated offline and stored as an adjaceny list in **Redis**.
*   **Time Complexity**: $O(1)$ lookup via Hash Set (`HGET`).

### 2. Item2Vec Semantic Embeddings
*   **Problem**: Co-occurrence suffers from the "Long Tail" problem (rarely purchased items never surface).
*   **Solution**: We migrated Word2Vec (Skip-Gram) modeling from NLP to E-Commerce. By treating a historical "User Cart" as a "Sentence" and each "Item" as a "Word", we learn 64-dimensional latent representations of our catalog.
*   **Deployment**: Using `gensim`, the model clusters items logically. A newly added 'Sizzling Brownie' will map to the exact same vector neighborhood as existing desserts, allowing it to be natively recommended without zero-interaction history (Cold Start mitigation).

---

## Layer 2: Feature Assembly (The 22-Dimensional Vector)
Once the candidate pool is reduced to 50 items, the FastAPI server constructs a 22-dimensional feature matrix for each candidate. These features fall into three buckets:

### 1. User Context (Demographics & Elasticity)
*   **`avg_order_value`**: Defines the user's economic tier.
*   **`is_veg_user`**: Imputed historical vegetarian alignment.
*   **`price_elasticity_score`**: Calculates if a user abandons carts when recommended items exceed $30\%$ of the base total.

### 2. Item Context (Profit & Popularity)
*   **`popularity_percentile`**: Raw transaction counts grouped temporally.
*   **`margin_estimate`**: Hardcoded tiering evaluating the theoretical profit margin for Zomato.

### 3. Cart-State Context (The Most Critical Layer)
*   **`meal_stage`**: An explicit state machine tracking the Cartesian components of a meal: `{Main, Side, Drink, Dessert}`. Is a drink missing? Is a bread missing?
*   **`price_deviation_from_user_avg`**: Delta distance between candidate price and user's historical AOV.

---

## Layer 3: Machine Learning Ranking (LightGBM)
The aggregated feature matrix ($50 \text{ rows} \times 22 \text{ cols}$) is passed to the ML scoring layer.

### Why LightGBM?
We explicitly rejected Deep Learning sequential models (e.g., BERT4Rec, SASRec). Deep Learning requires GPU matrix multiplication to maintain latency SLAs. **LightGBM** (Gradient Boosted Decision Trees) executes purely on CPU, compiling histograms to traverse thousands of tree branches in $< 20$ms.

### LambdaRank Objective (NDCG)
Crucially, the objective function is **NOT** `binary_logloss`. A model predicting standard $0.0$ to $1.0$ logistic probability doesn't understand UX design. Instead, we use `lambdarank`.
LambdaRank models the explicitly sorted order natively, directly optimizing for **NDCG@8** (Normalized Discounted Cumulative Gain). It assigns much heavier mathematical punishment if it places a highly relevant item outside the visible Top-8 scrolling rail mechanism.

---

## Layer 4: Deterministic Business Re-Ranker
Models optimize statistical relevancy; they do *not* understand brand UX fatigue, cannibalization, or Gross Merchandise Value (GMV). We apply a deterministic $O(N \log N)$ sorting layer to the ML outputs to explicitly generate revenue.

### 1. The Diversity Sliding Window
*   **Constraint**: Maximum $2$ items per category in the Top 8. 
*   **Logic**: If the UI shows 8 different types of Naan, the user suffers decision-paralysis. The Re-Ranker explicitly walks the sorted ML array and deletes the 3rd instance of any category, pulling up diverse alternatives from the rear.

### 2. High-Margin Upsell Sequence
*   **Constraint**: Desserts have the highest profit margins for delivery platforms.
*   **Logic**: If the `Cart-State` feature reads that the meal is $\ge 50\%$ complete (has Main + Side), the system applies a $\gamma = 1.5$ multiplier to the Logit scores of all Desserts in the array, actively overriding the ML model to drive raw profitability.

### 3. Redundancy / Cannibalization Penalties
*   **Constraint**: Do not drive AOV at the cost of cart abandonment.
*   **Logic**: Recommending a heavy Side dish when a user *already* has a Side dish causes price shock. We apply a $\gamma = 0.5$ penalty to candidates if their category is currently overlapping heavily with the cart payload.

---

## Conclusion
This architecture actively answers Problem Statement 2 constraints by balancing the predictive, personalized power of ML semantic embeddings with the hyper-fast, brutally deterministic reality of an E-Commerce rail.
