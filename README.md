# Online Deal Scouter: Autonomous Multi-Agent Deal Hunting & Price Prediction Platform

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Fine-Tuned Model](https://img.shields.io/badge/HuggingFace-Llama--3.2--3B--QLoRA-orange.svg)](https://huggingface.co/Ashkillzz/price-2026-01-05_05.50.20-lite)
[![Dataset](https://img.shields.io/badge/HuggingFace-Curated--20K--Dataset-yellow.svg)](https://huggingface.co/datasets/Ashkillzz/curated_product_reviews)
[![Modal Serverless](https://img.shields.io/badge/Deployment-Modal_GPU-green.svg)](https://modal.com)
[![Weights & Biases](https://img.shields.io/badge/W%26B-Experiment_Tracking-black.svg)](https://wandb.ai)

**Online Deal Scouter** is an end-to-end autonomous retail intelligence system that discovers online product deals in real time, predicts their true fair market value using a hybrid ensemble of fine-tuned and frontier models, calculates discount margins, and pushes instant alerts for high-value arbitrage opportunities.

---

## Architecture Overview

```
                      ┌────────────────────────────────────────┐
                      │    Live E-Commerce Feeds (DealNews)    │
                      └──────────────────┬─────────────────────┘
                                         │ RSS Ingestion
                                         ▼
                      ┌────────────────────────────────────────┐
                      │             Scanner Agent              │
                      │  (GPT-4o-mini + Structured Outputs)    │
                      └──────────────────┬─────────────────────┘
                                         │ Filtered Deal Candidates
                                         ▼
                      ┌────────────────────────────────────────┐
                      │             Planning Agent             │
                      │        (Orchestrator & Filter)         │
                      └──────────────────┬─────────────────────┘
                                         │ Dispatch for Valuation
                                         ▼
                      ┌────────────────────────────────────────┐
                      │             Ensemble Agent             │
                      │         (Meta-Learner Blending)        │
                      └───────┬──────────┬──────────┬──────────┘
                              │          │          │
         ┌────────────────────┘          │          └────────────────────┐
         ▼                               ▼                               ▼
┌──────────────────┐           ┌──────────────────┐           ┌──────────────────┐
│ Specialist Agent │           │  Frontier Agent  │           │RandomForest Agent│
│ Fine-Tuned LLaMA │           │  ChromaDB RAG +  │           │Local MiniLM +    │
│  (Modal T4 GPU)  │           │   GPT-4o-mini    │           │Scikit-Learn      │
└────────┬─────────┘           └─────────┬────────┘           └────────┬─────────┘
         │                               │                             │
         └────────────────────┬──────────┴─────────────────────────────┘
                              │ Predicted Fair Market Price
                              ▼
                      ┌────────────────────────────────────────┐
                      │     Discount Evaluation & Filter       │
                      │      (Margin = Estimate - Price)       │
                      └──────────────────┬─────────────────────┘
                                         │ If Discount > $50
                                         ▼
                      ┌────────────────────────────────────────┐
                      │            Messaging Agent             │
                      │     (Pushover Push / Twilio SMS)       │
                      └──────────────────┬─────────────────────┘
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 ▼                                               ▼
┌─────────────────────────────────┐             ┌─────────────────────────────────┐
│     PostgreSQL Database         │             │      Gradio Web Dashboard       │
│  (Persistent Storage & Pooling) │             │   (3D Latent Space & Monitoring)│
└─────────────────────────────────┘             └─────────────────────────────────┘
```

---

## Key Features

1. **Domain-Specific Q-LoRA Fine-Tuning**:
   - Fine-tuned `meta-llama/Llama-3.2-3B` in 4-bit NormalFloat (`nf4`) precision on 20,000 curated Amazon retail records ([Ashkillzz/curated_product_reviews](https://huggingface.co/datasets/Ashkillzz/curated_product_reviews)).
   - Achieved **1.2475 evaluation loss** and **76.3% token accuracy**, hosted on Hugging Face ([Ashkillzz/price-2026-01-05_05.50.20-lite](https://huggingface.co/Ashkillzz/price-2026-01-05_05.50.20-lite)).

2. **Serverless Cloud GPU Serving (Modal)**:
   - High-throughput, low-latency model inference deployed on serverless NVIDIA Tesla T4 GPUs via Modal.
   - Built-in persistent volume caching (`hf-hub-cache`) and scale-to-zero economics.

3. **Hybrid Ensemble Pricing Architecture**:
   - Combines three specialized estimators:
     - **Specialist Agent**: Remote fine-tuned LLaMA-3.2 model.
     - **Frontier Agent**: RAG pipeline retrieving 5 nearest-neighbor items from ChromaDB (`all-MiniLM-L6-v2`) injected into GPT-4o-mini context.
     - **Random Forest Agent**: Fast, deterministic local baseline.
   - Blended via a meta-regressor, outperforming all single models by **38%–51%** in Mean Absolute Error ($R^2 = 0.8294$).

4. **Autonomous Deal Scanning & Structured Extraction**:
   - Continuous ingestion of DealNews RSS feeds across multiple consumer product categories.
   - Strict Pydantic JSON schema generation ensuring zero hallucinations on prices, URLs, and descriptions.

5. **Production Storage & Interactive Dashboard**:
   - **PostgreSQL Persistence**: SQLAlchemy ORM with connection pooling, transaction isolation, and B-tree indexing on discounts and prices.
   - **Gradio UI**: Real-time deals monitor, streaming execution logs via thread-safe queues, and an interactive 3D t-SNE projection of the embedding vector store.

---

## Model Benchmark & Evaluation

Tested on a held-out benchmark of 250 diverse retail products using Root Mean Squared Logarithmic Error (RMSLE), Mean Absolute Error (MAE), Mean Squared Error (MSE), and Hit Rate (predictions within \$40 or <20% relative error):

| Model / Architecture | MAE (Avg Error) | RMSLE | MSE | $R^2$ | Hit Rate |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Ensemble Meta-Learner (Winner)** | **\$49.37** | **0.4834** | **6,199.26** | **0.8294** | **69.2%** |
| **Fine-Tuned Llama-3.2-3B (Specialist)** | **\$79.25** | **0.6184** | **16,574.95** | **0.5439** | **56.8%** |
| **Base Zero-Shot GPT-4o-mini** | \$79.58 | 0.5849 | 21,114.30 | 0.4189 | 52.0% |
| **Fine-Tuned GPT-4o-mini** | \$91.45 | 0.6813 | 18,314.12 | 0.4960 | 44.0% |
| **Random Forest Regressor** | \$101.40 | 0.9390 | 18,112.57 | 0.5016 | 30.8% |
| **Gradient Boosting / XGBoost** | \$121.77 | 1.0119 | 26,307.41 | 0.2760 | 21.2% |
| **Human Estimator Benchmark** | \$126.55 | 1.0018 | 36,664.31 | -0.0090 | 32.0% |

> **Key Result**: The Ensemble Meta-Learner slashed MAE by **38%** over fine-tuned LLaMA-3.2 and **51%** over Random Forest, cutting MSE by **63%** and boosting $R^2$ to **0.8294**.

---

## Repository Structure

```
online-deal-scouter/
├── agents/                       # Autonomous Multi-Agent Hierarchy
│   ├── __init__.py
│   ├── agent.py                  # Abstract base agent with color-coded logging
│   ├── planning_agent.py         # Main orchestrator
│   ├── scanner_agent.py          # RSS feed reader with OpenAI Structured Outputs
│   ├── specialist_agent.py       # Modal RPC interface for fine-tuned LLaMA
│   ├── frontier_agent.py         # ChromaDB RAG + GPT-4o-mini estimator
│   ├── random_forest_agent.py    # Local MiniLM + Scikit-Learn regressor
│   ├── ensemble_agent.py         # Meta-regression blend of all estimators
│   ├── messaging_agent.py        # Pushover / Twilio alert dispatcher
│   └── deals.py                  # Deal and opportunity data models (Pydantic)
│
├── training/                     # Fine-Tuning Pipeline
│   ├── train_qlora.py            # Q-LoRA training script (Unsloth / TRL / PEFT)
│   ├── data_curation.py          # Tokenizer-aware cleaning and prompt generation
│   └── export_adapter.py         # LoRA weight merging and Hugging Face upload
│
├── services/                     # Cloud Deployment
│   ├── pricer_service.py         # Modal serverless GPU deployment definition
│   └── keep_warm.py              # Optional container warm-up utility
│
├── database/                     # Persistence Layer
│   ├── connection.py             # SQLAlchemy engine with connection pooling
│   └── models.py                 # Deal and Opportunity ORM schemas
│
├── app.py                        # Gradio interactive web dashboard
├── deal_agent_framework.py       # Pipeline execution controller
├── seed_vectorstore.py           # ChromaDB vector store seeding script
├── train_baselines.py            # Random Forest baseline and Ensemble training
├── test_live_pipeline.py         # End-to-end live pipeline verification test
├── evaluate_models.py            # Benchmark evaluation CLI
├── items.py                      # Data schema and tokenizer utilities
├── testing.py                    # Benchmarking and evaluation harness
├── requirements.txt              # Production dependencies
└── .env.example                  # Environment variable template
```

---

## Quickstart Guide

### 1. Prerequisites & Environment Setup
Clone the repository and install dependencies:

```bash
git clone https://github.com/Ashkillzz/Online-Deal-Scouter.git
cd Online-Deal-Scouter
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file from the template:

```bash
cp .env.example .env
```

Set the required environment keys in `.env`:
```ini
OPENAI_API_KEY=sk-...
HF_TOKEN=hf_...
MODAL_TOKEN_ID=...
MODAL_TOKEN_SECRET=...
DATABASE_URL=postgresql://user:password@localhost:5432/deal_scout
PUSHOVER_USER=...       # Optional: For push alerts
PUSHOVER_TOKEN=...      # Optional: For push alerts
```

---

### 2. Deploy Fine-Tuned Model to Modal Serverless GPU
Deploy the remote inference service with automatic GPU provisioning:

```bash
modal deploy services/pricer_service.py
```

The deployed service will be registered under the app name `pricer-service` and automatically scale to zero when idle.

---

### 3. Initialize the Vector Store, Baseline Models & Database
Seed the ChromaDB vector store with reference products, train the local Random Forest & Ensemble models, and initialize database tables:

```bash
# 1. Initialize PostgreSQL / SQLite tables
python -c "from database.connection import init_db; init_db()"

# 2. Seed ChromaDB vector store (1,500 reference products)
python seed_vectorstore.py

# 3. Train local Random Forest baseline & Ensemble meta-regressor
python train_baselines.py

# 4. Optional: Run end-to-end smoke test
python test_live_pipeline.py
```

---

### 4. Run the Application
Launch the live deal hunter and Gradio monitoring dashboard:

```bash
python app.py
```

Access the dashboard at `http://localhost:7860`:
* Inspect live deals and discount margins.
* Explore the 3D t-SNE projection of the retail embedding space.
* Monitor real-time logs from the background agent execution queue.
* Click on any deal row to manually trigger a push alert notification.

---

## Weights & Biases Fine-Tuning Specifications

| Parameter | Configuration |
| :--- | :--- |
| **Run Path** | `ashkillzz-vellore-institute-of-technology/price/ygu7d083` |
| **Base Model** | `meta-llama/Llama-3.2-3B` (`3,231,099,904` params) |
| **Hardware** | 1× NVIDIA Tesla T4 (16 GB VRAM, CUDA 12.4) |
| **Quantization** | 4-bit NormalFloat (`nf4`), double quantization |
| **LoRA Config** | $r=32$, $\alpha=64$, dropout=0.1, target modules: `[q_proj, k_proj, v_proj, o_proj]` |
| **Training Setup** | 1 epoch, batch size 32, 625 steps, learning rate $1 \times 10^{-4}$ (cosine decay) |
| **Training Time** | 3,582 seconds (~59 mins 42s) |
| **Output Adapter** | [`Ashkillzz/price-2026-01-05_05.50.20-lite`](https://huggingface.co/Ashkillzz/price-2026-01-05_05.50.20-lite) |

---

## License

MIT License. See [LICENSE](LICENSE) for details.
