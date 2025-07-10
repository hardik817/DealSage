# 🛍️ DealSage – The Price is Right

An AI-powered platform that autonomously scrapes online product deals, filters high-value items, estimates their actual price using an ensemble of agents, and visualizes deal insights — in real-time.

> 💡 Combines cutting-edge LLMs, RAG, fine-tuning, and classical ML for a production-ready pricing agent system.

---

## 🚀 Features

- 🔍 **Autonomous Deal Scraping** from RSS feeds via `feedparser` and `BeautifulSoup`
- 🤖 **Multi-Agent Pricing Architecture**:
  - 🧠 `SpecialistAgent`: Fine-tuned Meta LLaMA 3.1 (deployed via Modal)
  - 🔍 `FrontierAgent`: GPT-4o or Gemini 2.5 with RAG via ChromaDB
  - 📄 `ScannerAgent`: Gemini 2.5 Flash with JSON-structured parsing
  - 🌲 `RandomForestAgent`: ML model with MiniLM sentence embeddings
  - 🧮 `EnsembleAgent`: Combines predictions via Linear Regression
- 📊 **Live Gradio Interface** with:
  - Real-time logging
  - 3D semantic embeddings (Plotly)
- 🧠 **Memory + Planning System** to avoid duplicates and ensure continuous improvement
- 📡 **Push Notification & Alerts** for high-discount opportunities

---

## 🧪 Training Logs (Weights & Biases)

Model training performance was monitored using [Weights & Biases](https://wandb.ai), tracking convergence, metrics, and stability:

![Training Loss Curve](https://github.com/hardik817/DealSage/blob/main/assets/image.png)
*Training Loss Curve*


> 📌 Fine-tuned on LLaMA 3.1 with 4-bit quantization using `bitsandbytes`, `accelerate`, and `peft`.

---

## 🧪 Working Demo

The system autonomously identifies and evaluates deals using multiple AI agents, visualizing logs and predictions:

![Working Demo Screenshot](https://github.com/hardik817/DealSage/blob/main/assets/Screenshot%202025-07-10%20083419.png)
![Notifications](https://github.com/hardik817/DealSage/blob/main/assets/WhatsApp%20Image%202025-07-10%20at%2009.21.35_89d74b06.jpg)

---
*Agent-based system identifying product deals with UI logging and analytics*

### 🔁 Live Log Panel

![Live Logs](https://github.com/hardik817/DealSage/blob/main/assets/Screenshot%202025-07-10%20083532.png)
*Real-time collaboration and prediction logs*

> 🧠 Observe agent interactions, predictions, and decision-making — all streamed live.

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
