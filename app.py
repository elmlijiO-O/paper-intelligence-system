from flask import Flask, jsonify, request, render_template
import json
import os
from modules import search as search_mod

app = Flask(__name__)

# Seed FAISS index at startup
def _seed_index():
    path = os.path.join(os.path.dirname(__file__), "mock_data", "sample_papers.json")
    if os.path.exists(path):
        with open(path) as f:
            papers = json.load(f)
        search_mod.build_index(papers)
        print(f"[startup] FAISS index built with {len(papers)} papers.")
    else:
        print("[startup] WARNING: mock_data/sample_papers.json not found.")

_seed_index()

# ── HTML routes ──────────────────────────────────────────────────────────────
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

# ── API routes ───────────────────────────────────────────────────────────────
@app.route("/api/search")
def api_search():
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify({"error": True, "message": "Missing required param: q", "code": 400}), 400
    limit = int(request.args.get("limit", 10))
    if not search_mod.is_ready():
        return jsonify({"error": True, "message": "Index loading, try again in 30s.", "code": 503}), 503
    try:
        results = search_mod.search(q, k=limit)
        return jsonify({"query": q, "results": results, "count": len(results)})
    except Exception as e:
        return jsonify({"error": True, "message": str(e), "code": 500}), 500

@app.route("/api/paper/<arxiv_id>")
def api_paper(arxiv_id):
    from modules.fetcher import fetch_paper_by_id

    from modules.nlp import summarize, extract_keywords
    from modules.analysis import classify_section, explain_concepts
    paper = fetch_paper_by_id(arxiv_id)
    if paper is None:
        return jsonify({"error": True, "message": f"Paper with ID {arxiv_id} not found on arXiv.", "code": 404}), 404
    abstract = paper["abstract"]
    try:
        summary = summarize(abstract)
        keywords = extract_keywords(abstract, n=10)
    except Exception as e:
        return jsonify({"error": True, "message": f"NLP error: {e}", "code": 500}), 500
    try:
        section_label = classify_section(abstract)
        concepts = explain_concepts(abstract)
    except Exception:
        section_label = "unknown"
        concepts = []
    return jsonify({"paper": paper, "summary": summary, "keywords": keywords, "section_label": section_label, "concepts": concepts})

@app.route("/api/attention/<arxiv_id>")
def api_attention(arxiv_id):
    from modules.fetcher import fetch_paper_by_id
    from modules.analysis import get_attention
    paper = fetch_paper_by_id(arxiv_id)
    if paper is None:
        return jsonify({"error": True, "message": f"Paper with ID {arxiv_id} not found on arXiv.", "code": 404}), 404
    try:
        result = get_attention(paper["abstract"])
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": True, "message": str(e), "code": 500}), 500

@app.route("/api/compare")
def api_compare():
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify({"error": True, "message": "Missing required param: q", "code": 400}), 400
    from modules.compare import compare
    path = os.path.join(os.path.dirname(__file__), "mock_data", "sample_papers.json")
    with open(path) as f:
        papers = json.load(f)
    try:
        result = compare(q, papers)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": True, "message": str(e), "code": 500}), 500

if __name__ == "__main__":
    app.run(debug=True)