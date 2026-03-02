# ZomaThon: ML Model Architecture & Funnel

This diagram exclusively outlines the data science logic isolating Candidate Generation from the Scoring execution. A key priority in recommendation engines is handling class imbalance efficiently; we accomplish this by using a strictly layered funnel model ending in hard business constraints.

## The 4-Layer Funnel Logic

![ML Model Architecture](assets/modelarchitecture.png)

### Explanatory Layers
1. **Layer 1 (Candidate Generation)**: The model prevents global scale memory exhaustion by pulling a localized subset ($K \approx 50$) of semantically associated candidate strings through Redis indexing paths rather than predicting across the entire restaurant catalog inline. 
2. **Layer 2 & 3 (Feature Scoring via LightGBM)**: Extracts temporal logic alongside historical purchase trends. Instead of predicting a flat boolean score (LogLoss classification pattern), LambdaRank actively computes relative order logic directly modeling UX placement (NDCG).
3. **Layer 4 (Business Constraints)**: An deterministic execution environment prioritizing actual Zomato unit-economics, ensuring a diverse slate that natively boosts high-margin items once a cart reaches an identified "completion milestone".
