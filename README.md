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

