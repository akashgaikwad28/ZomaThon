<p align="center">
  <img src="assets/zomathon_logo.png" width="800" alt="Zomathon Logo"/>
</p>

<p align="center">
  <b>Cart Super Add-On (CSAO) Recommendation Engine</b>
</p>

# ZomaThon: Intelligent Cart Recommendations

**ZomaThon CSAO** is a production-ready, ultra-low latency recommendation engine designed for Zomato's Problem Statement 2 constraints. It transforms active, potentially incomplete user carts in real-time and predicts the optimal add-on items to complete the meal, driving Average Order Value (AOV) while strictly maintaining a sub-250ms P99 latency SLA.

---

## 🏗️ System Architecture

The system is built on a **Hybrid 4-Layer Funnel Pipeline** architecture:
1. **Candidate Retrieval Layer ($O(1)$)**: Rapid candidate mining reducing the restaurant's entire menu to ~50 items using pre-computed co-occurrence and Item2Vec semantic embeddings mapped in Redis.
2. **ML Ranking (LambdaRank)**: Contextual scoring of candidates based on 22 engineered features utilizing Gradient Boosted Trees (LightGBM).
3. **Business Re-Ranker**: A deterministic rule-engine enforcing sequence state logic (e.g., punishing redundant sides, mathematically boosting high-margin desserts).
4. **AI Transparency**: An asynchronous LLM caching layer (Gemini/Groq) providing the "Why" behind the top recommendation.

*(View our interactive architecture diagrams in the `assets/` directory!)*

---

## ✨ Core Features

### 1. The "Incomplete Meal" State Tracker
The system natively understands context. If a user orders a Biryani and a Coke, the system recognizes the meal is missing a side/dessert and dynamically boosts those categories while down-weighting extra mains.

### 2. LightGBM LambdaRank Optimization
Instead of optimizing for global classification (LogLoss), the system uses LambdaRank to directly optimize for **Normalized Discounted Cumulative Gain (NDCG@8)**, ensuring the most relevant items appear at the very front of the UI rail. 

### 3. Strict Business Guardrails
Models lack business sense. Our deterministic Layer 3 enforces:
* **Diversity**: No more than 2 items from the same category.
* **Cannibalization Safety**: Aggressive penalties for suggesting a Side when a Side is already in the cart.
* **Margin Optimization**: Up-sell multipliers for high-margin Desserts & Beverages when a meal is $\ge 50\%$ complete.

### 4. Generative AI Explainability
Uses **Gemini 2.0 Flash** & **Groq (Llama 3.3)** asynchronously to generate human-readable explanations ("Perfectly balances the spice of your Biryani") cached in Redis for <5ms execution times.

---

## 🛡️ Multi-LLM Resilience Strategy
Designed for mission-critical reliability, ZomaThon implements a tiered failover strategy to protect the <250ms latency SLA:
*   **Primary**: **Gemini 2.0 Flash** (High reasoning).
*   **Secondary**: **Groq Llama 3.3** (High-speed fallback if Gemini hits rate limits).
*   **Tertiary**: **Local Cache / Static Fallbacks** (Emergency extraction if all cloud APIs are unavailable).

---

## 🛠 Tech Stack
*   **Languages**: Python 3.9+
*   **Machine Learning**: LightGBM, Gensim (Word2Vec), Scikit-Learn.
*   **Infrastructure**: FastAPI, Uvicorn, Docker Compose.
*   **Database & Cache**: PostgreSQL (Feature Store), Redis (Candidate & LLM Caching).

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- Docker & Docker Compose
- `pip` or `uv` package manager

### Installation & Setup
1. **Clone the Project**:
   ```bash
   git clone <repository-url>
   cd ZomaThon
   ```
2. **Configure Environment**:
   Copy the example environment file and fill in API keys:
   ```bash
   cp .env.example .env
   # Add your GEMINI_API_KEY and GROQ_API_KEY inside .env
   ```
3. **Install Dependencies**:
   ```bash
   pip install -r infra/requirements.txt
   ```

### Running the System (From Scratch)
Since model weights (`.pkl`) and data files (`.parquet`) are explicitly `.gitignore`d to keep the repository clean, you must execute the offline training pipeline once before starting the API.

**Step 1. Boot up the infrastructure**
```bash
# Boots Postgres (Feature Store) and Redis (Cache)
docker-compose up -d
```

**Step 2. Generate Synthetic Data & Preprocess**
```bash
# Generates 10,000 orders and mathematically constraints them
python training/data_generation/advanced_data_generator.py

# Assembles the User/Item/Cart PostgreSQL Feature Matrix
python training/preprocessing/preprocess.py
```

**Step 3. Train Models & Build Indexes**
```bash
# Build the Candidate Retrieval layers
python app/retrieval/co_occurrence.py
python training/train_item2vec.py

# Train the primary LightGBM LambdaRank model
python training/train_lgbm.py
```

**Step 4. Start the Inference API**
```bash
# Boot the FastAPI Server
python -m app.main
```
*Access the interactive Swagger UI at: `http://localhost:8000/docs`*

### Verification & Benchmarking
```bash
# Verify the strict <250ms P99 SLA constraint
python scripts/benchmark.py

# Evaluate Offline Metrics (NDCG@8, Precision@8, AOV Lift)
python training/evaluate.py
```

---

## 📁 Repository Structure
```text
ZomaThon/
├── app/         # FastAPI, Retrieval, Ranking, Business Rules, LLM Services
├── assets/      # Interactive HTML Architecture Diagrams
├── data/        # Synthetic Parquet Datasets (Raw & Processed)
├── infra/       # Docker Compose, Dockerfiles, Requirements
├── notebooks/   # Data Science workflows (EDA to Model Training)
├── scripts/     # API Load Benchmarking
├── tests/       # PyTest evaluation suites
└── training/    # Advanced Data Generator, Preprocessing, LGBM & Item2Vec Training
```

---

## 📚 Documentation
To observe the exploratory workflows, modeling iterations, and massive engineering rigor that led to this pipeline, review the following:

**Interactive Architecture Diagrams:**
- [Deployment System Architecture](system_architecture.md)
- [ML Model Funnel Architecture](model_architecture.md)
- [End-to-End Workflow](workflow.md)

**Data Science Notebooks (`notebooks/`):**
- `01_data_exploration.ipynb`
- `02_feature_engineering.ipynb`
- `03_model_training.ipynb`
- `04_item2vec_embeddings.ipynb`
- `05_synthetic_data_generation.ipynb`
- `06_llm_explanations.ipynb`

**Final Submission Artifacts:**
Please see the generated technical dossiers in your local environment detailing the mathematics of the data pipeline, the business re-ranker, and the LightGBM evaluation methodology.
