import json
import pathlib
from flask import Flask, jsonify, request, render_template
from modules.fetcher import fetch_papers, fetch_paper_by_id
from modules.nlp import summarize, extract_keywords
from modules.compare import compare
from modules.search import build_index, search
from modules.analysis import get_attention, classify_section, explain_concepts

app = Flask(__name__)

# ── Seed data ─────────────────────────────────────────────────
SEED_PAPERS = json.loads(
    pathlib.Path("mock_data/sample_papers.json").read_text()
)

# ── Build search index at startup ─────────────────────────────
print("[startup] Building FAISS index from seed papers...")
build_index(SEED_PAPERS)
print("[startup] Index ready.")

# ── HTML routes ───────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/paper/<arxiv_id>")
def paper_detail(arxiv_id):
    return render_template("paper.html", arxiv_id=arxiv_id)

@app.route("/attention/<arxiv_id>")
def attention_page(arxiv_id):
    return render_template("attention.html", arxiv_id=arxiv_id)

@app.route("/compare")
def compare_page():
    return render_template("compare.html")

# ── API routes ────────────────────────────────────────────────
@app.route("/api/search")
def api_search():
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify({"error": True, "message": "Missing query param 'q'", "code": 400}), 400
    limit = int(request.args.get("limit", 10))
    try:
        results = search(q, k=limit)
        if not results:
            results = fetch_papers(q, max_results=limit)
        if not results:
            results = SEED_PAPERS
    except Exception:
        try:
            results = fetch_papers(q, max_results=limit)
        except Exception:
            results = SEED_PAPERS
    return jsonify({"query": q, "results": results, "count": len(results)})


@app.route("/api/paper/<arxiv_id>")
def api_paper(arxiv_id):
    paper = fetch_paper_by_id(arxiv_id)
    if not paper:
        paper = next((p for p in SEED_PAPERS if p["id"] == arxiv_id), None)
    if not paper:
        return jsonify({"error": True, "message": f"Paper {arxiv_id} not found.", "code": 404}), 404

    try:
        summary = summarize(paper["abstract"])
    except Exception:
        summary = paper["abstract"][:300] + "..."
    try:
        keywords = extract_keywords(paper["abstract"])
    except Exception:
        keywords = []
    try:
        section_label = classify_section(paper["abstract"])
    except Exception:
        section_label = "unknown"
    try:
        concepts = explain_concepts(paper["abstract"])
    except Exception:
        concepts = []

    return jsonify({
        "paper": paper,
        "summary": summary,
        "keywords": keywords,
        "section_label": section_label,
        "concepts": concepts,
    })


@app.route("/api/attention/<arxiv_id>")
def api_attention(arxiv_id):
    paper = fetch_paper_by_id(arxiv_id)
    if not paper:
        paper = next((p for p in SEED_PAPERS if p["id"] == arxiv_id), None)
    if not paper:
        return jsonify({"error": True, "message": f"Paper {arxiv_id} not found.", "code": 404}), 404
    try:
        result = get_attention(paper["abstract"])
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": True, "message": str(e), "code": 500}), 500


@app.route("/api/compare")
def api_compare():
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify({"error": True, "message": "Missing query param 'q'", "code": 400}), 400
    limit = int(request.args.get("limit", 5))
    try:
        papers = fetch_papers(q, max_results=20)
        if not papers:
            papers = SEED_PAPERS
    except Exception:
        papers = SEED_PAPERS
    result = compare(q, papers, k=limit)
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)