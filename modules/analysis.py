"""
analysis.py — Person B
Three NLP functions built on HuggingFace models:
    get_attention(abstract)      → tokens + attention weight matrix
    classify_section(abstract)   → one of 4 section labels
    explain_concepts(abstract)   → top 3 terms with explanations
"""

import re

# ---------------------------------------------------------------------------
# Lazy model cache — each model loads once, then stays in memory
# ---------------------------------------------------------------------------
_bert_model = None
_bert_tokenizer = None
_zero_shot = None

SECTION_LABELS = ["methodology", "results", "background", "survey"]
MAX_TOKENS = 64


def _get_bert():
    global _bert_model, _bert_tokenizer
    if _bert_model is None:
        from transformers import BertModel, BertTokenizer
        _bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        _bert_model = BertModel.from_pretrained(
            "bert-base-uncased", output_attentions=True
        )
        _bert_model.eval()
    return _bert_tokenizer, _bert_model


def _get_zero_shot():
    global _zero_shot
    if _zero_shot is None:
        from transformers import pipeline
        _zero_shot = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
        )
    return _zero_shot


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_attention(abstract: str) -> dict:
    """
    Run abstract through bert-base-uncased, average attention across all
    heads on the last layer, truncate to MAX_TOKENS tokens.

    Returns:
        {
            "tokens":  ["the", "cat", ...],          # length N
            "weights": [[float, ...], ...],           # N x N, rows sum ~1.0
            "note":    "string"
        }
    """
    import torch

    tokenizer, model = _get_bert()

    encoding = tokenizer(
        abstract,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_TOKENS,
    )

    with torch.no_grad():
        outputs = model(**encoding)

    # attentions: tuple of (1, num_heads, seq_len, seq_len) per layer
    # We use the last layer, average over heads
    last_layer_attn = outputs.attentions[-1]          # (1, heads, seq, seq)
    avg_attn = last_layer_attn[0].mean(dim=0)         # (seq, seq)
    weights = avg_attn.numpy().tolist()

    token_ids = encoding["input_ids"][0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(token_ids)

    return {
        "tokens": tokens,
        "weights": weights,
        "note": (
            f"Averaged over all heads, last layer of bert-base-uncased. "
            f"Truncated to {len(tokens)} tokens. "
            "weights[i][j] = attention from token i to token j."
        ),
    }


def classify_section(abstract: str) -> str:
    """
    Zero-shot classify the abstract into one of SECTION_LABELS.

    Returns:
        One of: "methodology", "results", "background", "survey"
    """
    classifier = _get_zero_shot()
    result = classifier(abstract, candidate_labels=SECTION_LABELS)
    return result["labels"][0]


def explain_concepts(abstract: str) -> list[dict]:
    """
    Extract up to 3 noun-phrase candidates from the abstract,
    then use zero-shot to produce a one-sentence explanation for each.

    Returns:
        [{"term": str, "explanation": str}, ...]   — up to 3 items
    """
    candidates = _extract_noun_phrases(abstract)[:5]
    if not candidates:
        return []

    classifier = _get_zero_shot()
    explanations = []

    for term in candidates[:3]:
        # Frame as: which label best describes what this term IS?
        description_labels = [
            f"{term} is a mathematical or statistical method",
            f"{term} is a neural network architecture or component",
            f"{term} is a dataset or benchmark",
            f"{term} is an evaluation metric",
            f"{term} is a general machine learning concept",
        ]
        result = classifier(abstract, candidate_labels=description_labels)
        best_label = result["labels"][0]
        # Strip the term prefix to get a clean sentence
        explanation = best_label.replace(f"{term} is ", "").capitalize()
        explanation = f"{term.capitalize()} is {explanation}."
        explanations.append({"term": term, "explanation": explanation})

    return explanations


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_noun_phrases(text: str) -> list[str]:
    """
    Lightweight noun-phrase extraction using regex — no spaCy required.
    Targets lowercase multi-word technical terms likely to appear in NLP abstracts.
    """
    # Match 1-3 lowercase words that look like technical compound nouns
    pattern = r"\b([a-z]+(?:-[a-z]+)*(?:\s[a-z]+(?:-[a-z]+)*){0,2})\b"
    raw = re.findall(pattern, text)

    # Filter: keep only multi-word or hyphenated phrases, skip stop words
    stopwords = {
        "the", "a", "an", "of", "in", "on", "at", "to", "for", "and",
        "or", "but", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "this", "that", "these",
        "those", "we", "our", "its", "it", "with", "from", "by", "as",
        "not", "no", "which", "when", "where", "how", "such", "than",
        "more", "also", "both", "each", "can",
    }

    seen = set()
    results = []
    for phrase in raw:
        phrase = phrase.strip()
        words = phrase.split()
        if len(words) < 2 and "-" not in phrase:
            continue
        if phrase in stopwords or all(w in stopwords for w in words):
            continue
        if phrase not in seen:
            seen.add(phrase)
            results.append(phrase)

    return results