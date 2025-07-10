# DealSage
# 🛍️ The Price is Right – Autonomous Deal Evaluation System

An AI-powered platform that scrapes online product deals, filters the most promising ones, estimates their true prices using ensemble learning, and visualizes deal value – all autonomously.

> 💡 Combines LLMs, RAG, fine-tuning, and classical ML to build a production-ready pricing agent system.

---

## 🚀 Features

- 🔍 **Deal Scraping** from multiple RSS feeds using `BeautifulSoup` and `feedparser`
- 🧠 **LLM Agents**:
  - Fine-tuned LLaMA 3 on Modal (SpecialistAgent)
  - GPT-4o with RAG over ChromaDB (FrontierAgent)
  - Gemini 2.5 Flash with JSON-structured parsing (ScannerAgent)
- 🌲 **Traditional ML Agents**:
  - Random Forest with MiniLM embeddings
  - Ensemble agent that combines predictions via Linear Regression
- 📈 **Interactive UI** using Gradio for real-time logging and analysis
- 📡 **Push Notifications** and autonomous opportunity alerts
- 🧠 **Memory + Planning System** to avoid duplication and enable continuous monitoring

---

## 🧩 System Architecture

```text
                     ┌────────────────────────────┐
                     │     RSS Feeds (DealNews)   │
                     └────────────┬───────────────┘
                                  │
                         [Scraper + ScannerAgent]
                                  │
                    ┌────────────▼─────────────┐
                    │ Filtered & Parsed Deals  │
                    └────────────┬─────────────┘
                                 ▼
                        [Pricing Agents Layer]
 ┌────────────────────────────┬──────────────────────────────┬────────────────────────────┐
 │ SpecialistAgent            │ FrontierAgent                │ RandomForestAgent          │
 │ Modal LLaMA FT model       │ GPT-4o / Gemini + Chroma RAG │ RF model with MiniLM embed │
 └────────────┬───────────────┴────────────┬─────────────────┴────────────┬──────────────┘
                          └────────────────────────────┐
                          [EnsembleAgent: Linear Regression]
                                       │
                                [Planner + Memory]
                                       │
                          ┌────────────▼────────────┐
                          │   Gradio UI + Logging   │
                          └────────────┬────────────┘
                                       ▼
                             📡 User Notification System

## 🧪 Training Logs (Weights & Biases)

Here’s how the model performed during training using [Weights & Biases](https://wandb.ai):

<p align="center">
  <img src="assets/wandb_loss_curve.png" width="600" alt="Training Loss Curve"/>
</p>

<p align="center">
  <img src="assets/wandb_metrics.png" width="600" alt="Validation Metrics"/>
</p>

We monitored training, loss convergence, and validation accuracy using W&B's dashboard. The model was trained on LLaMA 3.1 with 4-bit quantization using `bitsandbytes` and `peft`.


## 🧪 Working Demo

The agent-based system identifies potential deals, estimates their price using multiple AI agents, and presents results with logs and 3D embeddings:

<p align="center">
  <img src="assets/ui_working.png" width="700" alt="Working Demo Screenshot"/>
</p>

### 🔁 Log Panel

<p align="center">
  <img src="assets/log_panel.png" width="700" alt="Live Logs"/>
</p>

The logs show the collaboration between agents, price predictions, and deal evaluations in real-time. You can also visualize semantic similarity in the 3D Plotly graph powered by `sentence-transformers` and `plotly`.

---

