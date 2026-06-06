from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Load models once at import time — never inside a function
print("[nlp] Loading summarization model...")
_summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
print("[nlp] Summarization model ready.")


def summarize(abstract: str) -> str:
    """
    Summarize a paper abstract using BART.
    Returns a single paragraph string.
    """
    if len(abstract.split()) < 30:
        return abstract

    max_input = 1024
    tokens = abstract.split()
    if len(tokens) > max_input:
        abstract = " ".join(tokens[:max_input])

    result = _summarizer(
        abstract,
        max_length=120,
        min_length=40,
        do_sample=False,
    )
    return result[0]["summary_text"]


def extract_keywords(abstract: str, n: int = 10) -> list[str]:
    """
    Extract top-n keywords from an abstract using TF-IDF.
    Works on a single document by treating each sentence as a doc.
    """
    sentences = [s.strip() for s in abstract.split(".") if len(s.strip()) > 5]

    if len(sentences) < 2:
        # fallback: split into word chunks
        words = abstract.split()
        sentences = [" ".join(words[i:i+10]) for i in range(0, len(words), 10)]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=200,
    )

    try:
        tfidf_matrix = vectorizer.fit_transform(sentences)
    except ValueError:
        return []

    scores = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
    feature_names = vectorizer.get_feature_names_out()

    top_indices = scores.argsort()[::-1][:n]
    keywords = [feature_names[i] for i in top_indices]

    return keywords