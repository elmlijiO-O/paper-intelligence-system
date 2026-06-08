/**
 * attention.js — Person B
 * Renders a token-token attention heatmap using Chart.js.
 * Requires Chart.js 4.x loaded globally (via CDN in base.html).
 *
 * Called by attention.html:  loadAttention(paperId)
 */

async function loadAttention(paperId) {
  const loading = document.getElementById("loadingState");
  const content = document.getElementById("attentionContent");
  const errorEl = document.getElementById("errorState");
  const errorMsg = document.getElementById("errorMsg");
  const meta = document.getElementById("attentionMeta");

  try {
    const resp = await fetch(`/api/attention/${paperId}`);
    const data = await resp.json();

    if (!resp.ok) {
      throw new Error(data.message || "Failed to fetch attention data.");
    }

    loading.classList.add("hidden");
    content.classList.remove("hidden");

    // Show note
    meta.textContent = data.note || "";

    renderHeatmap(data.tokens, data.weights);
  } catch (err) {
    loading.classList.add("hidden");
    errorEl.classList.remove("hidden");
    errorMsg.textContent = err.message;
  }
}

function renderHeatmap(tokens, weights) {
  const n = tokens.length;

  // Build flat dataset: Chart.js matrix plugin expects {x, y, v}
  // We'll use a custom implementation with a single dataset of colored cells
  // drawn via a plugin, since chart.js-matrix is not on the CDN.
  // Instead: use a scatter chart with square point styles sized to fill cells.

  const canvas = document.getElementById("attentionChart");
  const ctx = canvas.getContext("2d");

  // Find max weight for colour scaling
  let maxW = 0;
  for (const row of weights) {
    for (const v of row) {
      if (v > maxW) maxW = v;
    }
  }

  // Cell size in px
  const LABEL_W = 80;
  const CELL = Math.max(12, Math.min(28, Math.floor(600 / n)));
  const SIZE = LABEL_W + n * CELL;

  canvas.width = SIZE;
  canvas.height = SIZE;

  // Background
  ctx.fillStyle =
    getComputedStyle(document.body).getPropertyValue("--bg") || "#0d0d0d";
  ctx.fillRect(0, 0, SIZE, SIZE);

  // Draw cells
  for (let i = 0; i < n; i++) {
    // row = attending token
    for (let j = 0; j < n; j++) {
      // col = attended-to token
      const v = weights[i][j] / (maxW || 1);
      ctx.fillStyle = weightToColor(v);
      ctx.fillRect(LABEL_W + j * CELL, LABEL_W + i * CELL, CELL - 1, CELL - 1);
    }
  }

  // Token labels — x axis (top)
  ctx.fillStyle = "#aaa";
  ctx.font = `${Math.min(11, CELL - 1)}px DM Mono, monospace`;
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  for (let j = 0; j < n; j++) {
    ctx.save();
    ctx.translate(LABEL_W + j * CELL + CELL / 2, LABEL_W - 4);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText(tokens[j], 0, 0);
    ctx.restore();
  }

  // Token labels — y axis (left)
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  for (let i = 0; i < n; i++) {
    ctx.fillText(tokens[i], LABEL_W - 4, LABEL_W + i * CELL + CELL / 2);
  }

  // Colour legend
  drawLegend(ctx, SIZE);
}

/**
 * Map a normalised weight [0, 1] to a CSS colour string.
 * Uses a dark-to-bright amber palette that looks good on dark background.
 */
function weightToColor(v) {
  // Dark navy → amber → bright yellow
  const r = Math.round(v * 255);
  const g = Math.round(v * 180);
  const b = Math.round((1 - v) * 80);
  return `rgb(${r},${g},${b})`;
}

function drawLegend(ctx, canvasSize) {
  const W = 160,
    H = 14;
  const x = canvasSize - W - 16;
  const y = canvasSize - H - 24;

  // Gradient bar
  const grad = ctx.createLinearGradient(x, y, x + W, y);
  grad.addColorStop(0, weightToColor(0));
  grad.addColorStop(1, weightToColor(1));
  ctx.fillStyle = grad;
  ctx.fillRect(x, y, W, H);

  // Labels
  ctx.fillStyle = "#aaa";
  ctx.font = "10px DM Mono, monospace";
  ctx.textAlign = "left";
  ctx.fillText("low", x, y + H + 12);
  ctx.textAlign = "right";
  ctx.fillText("high", x + W, y + H + 12);
  ctx.textAlign = "center";
  ctx.fillText("attention", x + W / 2, y + H + 12);
}
