# Hugging Face NLP Lab

## 🧠 Overview
This project is a hands-on NLP engineering lab built using Hugging Face Transformers and Sentence Transformers.

It demonstrates how modern NLP systems are built using:
- Pretrained transformer models
- Sentiment analysis pipelines
- Sentence embeddings
- Modular software architecture

---

## 🚀 Features

### 1. Sentiment Analysis Service
- Uses Hugging Face pipeline API
- Classifies text as positive or negative
- Simple abstraction layer for model inference

### 2. Embedding Service
- Uses Sentence Transformers (`all-MiniLM-L6-v2`)
- Converts text into vector representations
- Computes semantic similarity between sentences

### 3. NLP Orchestrator System
- Combines multiple NLP components
- Runs sentiment + embedding analysis together
- Produces structured AI analysis output

---

## 🏗 Project Structure
```
src/
├── sentiment_service.py
├── embedding_service.py
├── mini_nlp_system.py
├── tokenizer_demo.py
├── embedding_demo.py
├── model_hub_demo.py
```
---

## 🧪 What I Learned
- How transformer models work in practice
- Difference between tokenization and embeddings
- How to build modular ML systems
- How to structure NLP pipelines like production systems
- How to use Hugging Face Model Hub effectively

---

## ▶️ How to Run

### 1. Install dependencies
```bash
pip install transformers sentence-transformers torch
2. Run NLP system
python -m src.mini_nlp_system
3. Run demos (optional)
python -m src.model_hub_demo
python -m src.tokenizer_demo
python -m src.embedding_demo
________________________________________
👤 Author
Alvaro Vega
Aspiring AI Engineer | Machine Learning Engineer | NLP & LLM Systems
GitHub: https://github.com/Javier-DataScience

