"""
test_routes.py — Person B
Integration tests for all Flask API routes.
Requires app.py to be importable and index to be seeded.
"""

import json
import pytest


@pytest.fixture(scope="module")
def client():
    """Create a Flask test client with the index pre-seeded."""
    from app import app
    from modules import search

    # Seed with minimal mock data so routes work without live arXiv calls
    mock_papers = [
        {
            "id": "1706.03762",
            "title": "Attention Is All You Need",
            "authors": ["Vaswani, A.", "Shazeer, N."],
            "abstract": (
                "We propose the Transformer, a model based solely on attention "
                "mechanisms for sequence transduction tasks like machine translation."
            ),
            "url": "https://arxiv.org/abs/1706.03762",
            "year": 2017,
            "categories": ["cs.CL", "cs.LG"],
        },
        {
            "id": "1810.04805",
            "title": "BERT: Pre-training of Deep Bidirectional Transformers",
            "authors": ["Devlin, J.", "Chang, M."],
            "abstract": (
                "BERT pre-trains deep bidirectional representations from unlabeled "
                "text by jointly conditioning on left and right context."
            ),
            "url": "https://arxiv.org/abs/1810.04805",
            "year": 2019,
            "categories": ["cs.CL"],
        },
    ]
    search.build_index(mock_papers)

    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


# ---------------------------------------------------------------------------
# /api/search
# ---------------------------------------------------------------------------

def test_search_returns_200(client):
    resp = client.get("/api/search?q=transformer")
    assert resp.status_code == 200


def test_search_response_schema(client):
    resp = client.get("/api/search?q=transformer")
    data = json.loads(resp.data)
    assert "query" in data
    assert "results" in data
    assert "count" in data
    assert isinstance(data["results"], list)


def test_search_missing_q_returns_400(client):
    resp = client.get("/api/search")
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# /api/paper/<id>
# ---------------------------------------------------------------------------

def test_paper_known_id_returns_200(client):
    resp = client.get("/api/paper/1706.03762")
    assert resp.status_code == 200


def test_paper_unknown_id_returns_404(client):
    resp = client.get("/api/paper/9999.99999")
    assert resp.status_code == 404


def test_paper_response_has_required_fields(client):
    resp = client.get("/api/paper/1706.03762")
    data = json.loads(resp.data)
    assert "paper" in data
    assert "summary" in data
    assert "keywords" in data


# ---------------------------------------------------------------------------
# /api/attention/<id>
# ---------------------------------------------------------------------------

def test_attention_returns_200(client):
    resp = client.get("/api/attention/1706.03762")
    assert resp.status_code == 200


def test_attention_response_schema(client):
    resp = client.get("/api/attention/1706.03762")
    data = json.loads(resp.data)
    assert "tokens" in data
    assert "weights" in data
    n = len(data["tokens"])
    assert len(data["weights"]) == n
    for row in data["weights"]:
        assert len(row) == n


def test_attention_unknown_id_returns_404(client):
    resp = client.get("/api/attention/9999.99999")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# /api/compare
# ---------------------------------------------------------------------------

def test_compare_returns_200(client):
    resp = client.get("/api/compare?q=transformer")
    assert resp.status_code == 200


def test_compare_response_schema(client):
    resp = client.get("/api/compare?q=transformer")
    data = json.loads(resp.data)
    assert "tfidf" in data
    assert "transformer" in data
    assert "overlap_count" in data


def test_compare_missing_q_returns_400(client):
    resp = client.get("/api/compare")
    assert resp.status_code == 400