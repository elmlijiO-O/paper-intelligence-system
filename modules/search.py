"""
search.py — Person B
Semantic search over paper abstracts using sentence-transformers + FAISS.
Usage:
    from modules.search import build_index, search
    build_index(papers)          # call once at startup
    results = search("query", k=10)
"""
 
import numpy as np
 
# Lazy imports — models are heavy, only load when first used
_model = None
_index = None
_papers = []  # master list, same order as FAISS index rows
 
 
def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model
 
 
def build_index(papers: list[dict]) -> None:
    """
    Encode all paper abstracts and store them in a FAISS flat index.
    Must be called before search().
 
    Args:
        papers: list of Paper objects (dicts with at least 'abstract' and 'id')
    """
    global _index, _papers
 
    import faiss
 
    if not papers:
        return
 
    model = _get_model()
    texts = [p.get("abstract", "") for p in papers]
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
 
    # L2-normalise so inner product == cosine similarity
    faiss.normalize_L2(embeddings)
 
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner Product (cosine after normalisation)
    index.add(embeddings)
 
    _index = index
    _papers = list(papers)
 
 
def search(query: str, k: int = 10) -> list[dict]:
    """
    Semantic search over the built index.
 
    Args:
        query: natural language query string
        k:     number of results to return
 
    Returns:
        Ranked list of Paper dicts, best match first.
 
    Raises:
        RuntimeError: if build_index() has not been called yet.
    """
    import faiss
 
    if _index is None:
        raise RuntimeError("FAISS index not yet built. Run build_index() first.")
 
    model = _get_model()
    q_vec = model.encode([query], convert_to_numpy=True, show_progress_bar=False)
    faiss.normalize_L2(q_vec)
 
    k = min(k, len(_papers))
    _scores, indices = _index.search(q_vec, k)
 
    results = []
    for idx in indices[0]:
        if idx == -1:
            continue
        results.append(_papers[idx])
 
    return results
 
 
def is_ready() -> bool:
    """Returns True if the index has been built and search is available."""
    return _index is not None