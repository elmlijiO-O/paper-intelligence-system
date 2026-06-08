# Research Paper Intelligence System

An AI-powered web application that retrieves, analyzes, and visualizes academic papers using a full NLP pipeline.

## Features

| Feature | Technique |
|---------|-----------|
| Semantic search | sentence-transformers + FAISS |
| Summarization | BART (facebook/bart-large-cnn) |
| Keyword extraction | TF-IDF (scikit-learn) |
| Attention visualization | BERT (bert-base-uncased) |
| Concept explanation | Zero-shot classification |
| Section classification | Zero-shot classification |
| Retrieval comparison | TF-IDF vs Transformer side-by-side |

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/paper-intelligence-system.git
cd paper-intelligence-system
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000` in your browser.

## Architecture
Flask (app.py)
├── modules/fetcher.py      arXiv API integration
├── modules/nlp.py          BART summarization + TF-IDF keywords
├── modules/compare.py      TF-IDF vs Transformer comparison
├── modules/search.py       Sentence-transformers + FAISS index
└── modules/analysis.py     BERT attention + zero-shot classification

## Project Structure
paper-intelligence-system/
├── app.py
├── modules/
├── templates/
├── static/
├── mock_data/
└── tests/

## Running Tests

```bash
python -m pytest tests/ -v
```

## Team

- Person A — NLP pipeline: summarization, keywords, retrieval comparison, arXiv fetcher
- Person B — NLP pipeline: semantic search, attention visualization, concept explainer, section classifier