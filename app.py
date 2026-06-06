from flask import Flask, jsonify, request, render_template

app = Flask(__name__)

# ── HTML routes ──────────────────────────────────────────────
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
    return jsonify({"status": "not implemented"})

@app.route("/api/paper/<arxiv_id>")
def api_paper(arxiv_id):
    return jsonify({"status": "not implemented"})

@app.route("/api/attention/<arxiv_id>")
def api_attention(arxiv_id):
    return jsonify({"status": "not implemented"})

@app.route("/api/compare")
def api_compare():
    return jsonify({"status": "not implemented"})

if __name__ == "__main__":
    app.run(debug=True)