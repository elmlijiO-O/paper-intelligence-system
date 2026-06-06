import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def compare(query: str, papers: list[dict], k: int = 5) -> dict:
    """
    Run the same query through TF-IDF and Transformer retrieval.
    Returns ranked results from both methods and their overlap.

    Note: Transformer retrieval here reuses cosine similarity on
    sentence-transformer embeddings. The search module handles
    embedding generation — we import its embed() utility.
    """
    if not papers:
        return _empty_response(query)

    abstracts = [p["abstract"] for p in papers]

    # ── TF-IDF retrieval ─────────────────────────────────────
    tfidf_results, tfidf_terms = _tfidf_search(query, papers, abstracts, k)

    # ── Transformer retrieval ─────────────────────────────────
    try:
        from modules.search import embed
        transformer_results, transformer_terms = _transformer_search(
            query, papers, abstracts, k
        )
    except Exception:
        transformer_results = []
        transformer_terms = []

    # ── Overlap ───────────────────────────────────────────────
    tfidf_ids = {p["id"] for p in tfidf_results}
    transformer_ids = {p["id"] for p in transformer_results}
    overlap_ids = list(tfidf_ids & transformer_ids)

    return {
        "query": query,
        "tfidf": {
            "results": tfidf_results,
            "top_terms": tfidf_terms,
        },
        "transformer": {
            "results": transformer_results,
            "top_terms": transformer_terms,
        },
        "overlap_count": len(overlap_ids),
        "overlap_ids": overlap_ids,
    }


def _tfidf_search(
    query: str, papers: list[dict], abstracts: list[str], k: int
) -> tuple[list[dict], list[str]]:
    corpus = abstracts + [query]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    try:
        tfidf_matrix = vectorizer.fit_transform(corpus)
    except ValueError:
        return [], []

    query_vec = tfidf_matrix[-1]
    doc_vecs = tfidf_matrix[:-1]

    sims = cosine_similarity(query_vec, doc_vecs).flatten()
    top_indices = sims.argsort()[::-1][:k]
    top_papers = [papers[i] for i in top_indices if sims[i] > 0]

    # top terms from query vector
    feature_names = vectorizer.get_feature_names_out()
    query_scores = np.asarray(query_vec.todense()).flatten()
    top_term_idx = query_scores.argsort()[::-1][:5]
    top_terms = [feature_names[i] for i in top_term_idx if query_scores[i] > 0]

    return top_papers, top_terms


def _transformer_search(
    query: str, papers: list[dict], abstracts: list[str], k: int
) -> tuple[list[dict], list[str]]:
    from modules.search import embed

    query_emb = embed([query])
    doc_embs = embed(abstracts)

    sims = cosine_similarity(query_emb, doc_embs).flatten()
    top_indices = sims.argsort()[::-1][:k]
    top_papers = [papers[i] for i in top_indices]

    # For transformer top terms: just return high-sim words from query
    words = [w for w in query.lower().split() if len(w) > 3]
    return top_papers, words


def _empty_response(query: str) -> dict:
    return {
        "query": query,
        "tfidf": {"results": [], "top_terms": []},
        "transformer": {"results": [], "top_terms": []},
        "overlap_count": 0,
        "overlap_ids": [],
    }