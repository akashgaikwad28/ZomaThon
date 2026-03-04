# ZomaThon: Deployment System Architecture (Deep Dive)

This document provides an exhaustive technical breakdown of the physical infrastructure and deployment topology of the ZomaThon Cart Super Add-On (CSAO) engine. The primary challenge addressed by this architecture is the strict **<250ms $P_{99}$ latency SLA** mandated by the problem statement, requiring structural separation of I/O-bound operations and CPU-bound ML execution.

## High-Level Infrastructure Topology

<iframe src="assets/system_architracture.png" width="100%" height="800px" style="border:none;"></iframe>

---

## 1. The Network Edge & Application Server

### Ingress & Load Balancing
All traffic from the Zomato client application arrives via an explicitly decoupled **API Gateway / Load Balancer** (e.g., Nginx or AWS ALB). 
*   **Why?**: Python web servers are notoriously bad at handling external TCP connection management and SSL/TLS termination. By offloading this to a reverse proxy, the internal application servers only deal with high-speed internal HTTP payloads.

### Uvicorn (ASGI) & FastAPI
The core compute layer runs on **FastAPI** wrapping **Uvicorn**, a lightning-fast ASGI (Asynchronous Server Gateway Interface) server built on `uvloop` (Cython).
*   **Concurrency Model**: Traditional WSGI servers (like Flask/Gunicorn) use process forking or threading, meaning 100 requests logically require 100 OS threads (heavy memory overhead). FastAPI utilizes Python's `asyncio` Event Loop. When the application issues a network request to the DB, the thread does not block; it yields context and handles incoming traffic. This allows a single worker to hold thousands of concurrent connections with minimal RAM.
*   **The GIL Bottleneck Strategy**: The Global Interpreter Lock (GIL) prevents True CPU parallelism in Python. Because ML Tree traversal (LightGBM) is strictly CPU-bound, a single prediction *will* momentarily block the Async Event loop. Therefore, the container is deployed with **Multi-Worker Uvicorn execution** (`--workers N` where $N = 2 \times \text{CPU Cores} + 1$), structurally preventing the SLA from failing during high throughput spikes.

---

## 2. In-Memory Computing Cache (Redis)

Standard relational databases are catastrophically slow for real-time recommendation retrieval ($>100$ms). We employ **Redis** as a distributed, in-memory caching layer that sits immediately adjacent to the FastAPI app.

### $O(1)$ Candidate Retrieval
*   The pre-calculated Co-occurrence arrays and Item2Vec neighbor graphs are flat-mapped into Redis as simple JSON strings keyed by `cand:{item_id}`.
*   **Latency Yield**: Fetching the initial 50 candidates requires a single `HGET` or `MGET` packet, completing network round-trips in **<4 milliseconds**. 

### GenAI Explanation Caching
*   LLM API calls (Gemini/Groq) routinely take $800\text{ms} - 1500\text{ms}$, which violently shatters our $250\text{ms}$ limit.
*   The explanation text is deterministically cached via `expl:{cart_hash}:{item_id}` allowing $99\%$ of requests to instantly return pre-computed human-readable text ($<2\text{ms}$).

---

## 3. Persistent Data Stores & ML Artifacts

### The Feature Store (PostgreSQL / Parquet)
For the LightGBM LambdaRank model to operate, it must dynamically assemble a 22-dimensional matrix based on User histories (Avg Order Value) and Item metadata (Base price, Margins).
*   **Topology**: Read-replicas of PostgreSQL or statically compiled Parquet data lakes are mounted directly into the container's disk or fetched via VPC endpoints.
*   **Updates**: Updated asynchronously via offline nightly ETL Cron jobs.

### Model Artifacts (`.pkl` weights)
The actual mathematical brains of the system—the compiled LightGBM tree histograms and the `gensim` semantic vector maps—are loaded securely into RAM once at Application Boot (`main.py` init cycle). 

---

## 4. Asynchronous Threads for Third-Party APIs

Because generating brand new explanations via **Gemini 2.0 Flash** or **Groq Llama-3.3** takes severe time, we physically detach this operation from the critical user-blocking path.
*   The system uses FastAPI's `BackgroundTasks` to fire a thread parallel to the HTTP response cycle. We return the core numerical recommendations to the user immediately, while the GenAI LLM resolves in the background and silently writes to Redis for the *next* time a user encounters that item.

---

## 5. Observability Stack (Datadog / ELK)

To prove to stakeholders that the $<250$ms SLA is met, rigorous telemetry is structurally integrated into the orchestration framework.
*   **Custom Middleware Latency Headers**: An `@app.middleware("http")` hook executes `time.time()` bounds around every individual HTTP request.
*   **Structured JSON Logging**: Python's native logger is hijacked to emit strict `JSON` formatted payloads: `{"latency_ms": 110, "path": "/recommend", "status": 200}`.
*   **Trace Extraction**: A sidecar container (e.g., FileBeat) continuously tails the log files, streaming the JSON directly into an ELK or Datadog dashboard for real-time percentile alerting. No brittle regex parsing required.
