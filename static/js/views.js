/* rendering helpers — canvas heatmaps + DOM builders (no frameworks) */

const VIRIDIS = [
  [68, 1, 84], [72, 40, 120], [62, 74, 137], [49, 104, 142],
  [38, 130, 142], [31, 158, 137], [53, 183, 121], [109, 205, 89],
  [180, 222, 44], [253, 231, 37],
];

export function viridis(v) {
  v = Math.max(0, Math.min(1, v));
  const t = v * (VIRIDIS.length - 1), i = Math.min(Math.floor(t), VIRIDIS.length - 2), f = t - i;
  const c = VIRIDIS[i].map((a, k) => Math.round(a + (VIRIDIS[i + 1][k] - a) * f));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

export function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text != null) e.textContent = text;
  return e;
}

export const $ = (id) => document.getElementById(id);

/* GPT-2 byte-BPE markers → readable display: Ġ = space before, Ċ = newline */
export function disp(t) {
  return String(t).replaceAll("Ġ", "·").replaceAll("Ċ", "↵");
}

/* -------- matrix heatmap on canvas with labels + hover -------- */
export function drawMatrix(canvas, matrix, opts = {}) {
  const { rowLabels = [], colLabels = [], shadeMask = null,
    cellW = 34, cellH = 24, onHover = null, vmax = null } = opts;
  const nR = matrix.length, nC = matrix[0].length;
  const labW = 74, labH = 62;
  canvas.width = labW + nC * cellW + 42;   // right pad: rotated labels lean up-right
  canvas.height = labH + nR * cellH;
  canvas.style.width = canvas.width + "px";
  const ctx = canvas.getContext("2d");
  const vm = vmax != null ? vmax : Math.max(...matrix.flat());
  ctx.fillStyle = "#ffffff"; ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.font = "10.5px ui-monospace, Menlo, monospace";
  for (let i = 0; i < nR; i++) {
    for (let j = 0; j < nC; j++) {
      const v = matrix[i][j];
      const blocked = shadeMask && shadeMask[i][j];
      ctx.fillStyle = blocked ? "#f3e7e7" : viridis(v / vm);
      ctx.fillRect(labW + j * cellW, labH + i * cellH, cellW - 1, cellH - 1);
      if (blocked) {
        ctx.strokeStyle = "#cc9999"; ctx.beginPath();
        const x = labW + j * cellW, y = labH + i * cellH;
        ctx.moveTo(x + 5, y + 5); ctx.lineTo(x + cellW - 6, y + cellH - 6);
        ctx.stroke();
      }
    }
  }
  ctx.fillStyle = "#5c6675"; ctx.textAlign = "right";
  for (let i = 0; i < nR; i++) ctx.fillText(rowLabels[i] ?? i, labW - 6, labH + i * cellH + cellH / 2 + 3);
  for (let j = 0; j < nC; j++) {
    ctx.save();
    ctx.translate(labW + j * cellW + cellW / 2 + 3, labH - 6);
    ctx.rotate(-Math.PI / 4);
    ctx.textAlign = "left"; ctx.fillStyle = colLabels[j]?.startsWith("[") ? "#9aa3b2" : "#5c6675";
    ctx.fillText(colLabels[j] ?? j, 0, 0);
    ctx.restore();
  }
  if (onHover) bindHover(canvas, ctx, matrix, { labW, labH, cellW, cellH, rowLabels, colLabels, shadeMask }, onHover);
}

function bindHover(canvas, ctx, matrix, g, onHover) {
  canvas.onmousemove = (e) => {
    const r = canvas.getBoundingClientRect();
    const x = e.clientX - r.left, y = e.clientY - r.top;
    const j = Math.floor((x - g.labW) / g.cellW), i = Math.floor((y - g.labH) / g.cellH);
    if (i >= 0 && i < matrix.length && j >= 0 && j < matrix[0].length)
      onHover(e.clientX, e.clientY, g.rowLabels[i], g.colLabels[j], matrix[i][j], g.shadeMask?.[i]?.[j]);
  };
  canvas.onmouseleave = () => onHover(0, 0, null);
}

/* -------- one-row stripe heatmap (192 sampled dims or 64 dims) -------- */
export function drawStripe(canvas, values, opts = {}) {
  const { labels = [], cell = 9, rowLabels = [] } = opts;
  const rows = values.length / 64 > 1.01 ? 0 : 1;   // 0 => grid already 2-D
  const is2D = Array.isArray(values[0]);
  const nR = is2D ? values.length : 1;
  const nC = is2D ? values[0].length : values.length;
  const labW = 34, labH = labels.length ? 18 : 6;
  canvas.width = labW + nC * cell;
  canvas.height = labH + nR * cell;
  canvas.style.width = canvas.width + "px";
  const ctx = canvas.getContext("2d");
  const all = (is2D ? values.flat() : values);
  const vmax = Math.max(...all.map(Math.abs), 1e-6);
  ctx.fillStyle = "#ffffff"; ctx.fillRect(0, 0, canvas.width, canvas.height);
  for (let i = 0; i < nR; i++) for (let j = 0; j < nC; j++) {
    const v = is2D ? values[i][j] : values[j];
    const m = Math.abs(v) / vmax;
    ctx.fillStyle = v >= 0 ? viridis(0.25 + 0.75 * m) : `rgb(${205 + 40 * m | 0},70,80)`;
    ctx.fillRect(labW + j * cell, labH + i * cell, cell - 1, cell - 1);
  }
  ctx.font = "10px ui-monospace, Menlo, monospace"; ctx.fillStyle = "#5c6675"; ctx.textAlign = "right";
  rowLabels.forEach((t, i) => ctx.fillText(t, labW - 5, labH + i * cell + cell / 2 + 3));
  return { labW, cell };
}

/* -------- shared small builders -------- */
export function chipRow(tokens, { special = [], sel = -1, onPick = null, sub = [] } = {}) {
  const box = el("div", "chips");
  tokens.forEach((t, i) => {
    const c = el("button", "chip" + (special[i] ? " special" : "") + (i === sel ? " sel" : ""), t);
    c.title = `position ${i}` + (sub[i] ? " · subword" : "");
    if (onPick) c.onclick = () => onPick(i);
    box.appendChild(c);
  });
  return box;
}

export function barColor(fam) { return fam === "bert" ? "var(--bert)" : "var(--gpt)"; }

export function stackedNorms(tokens, comps, fam) {
  /* comps: array of {label, color, norms:[.. per token ..]} */
  const box = el("div", "block");
  box.appendChild(el("div", "hint", "per-token L2 norms — stacked contributions"));
  const row = el("div", "stackbars");
  const totals = tokens.map((_, i) =>
    comps.reduce((a, c) => a + (c.norms?.[i] || 0), 0));
  const scale = 84 / Math.max(...totals, 0.01);   // bound the STACK, not one segment
  tokens.forEach((_, i) => {
    const b = el("div", "stackbar");
    comps.forEach(c => {
      const seg = el("div");
      seg.style.height = Math.max(c.norms[i] * scale, 0.4) + "px";
      seg.style.background = c.color;
      seg.title = `${c.label}: ${c.norms[i]}`;
      b.appendChild(seg);
    });
    row.appendChild(b);
  });
  box.appendChild(row);
  const legend = el("div", "hint");
  comps.forEach(c => {
    const d = el("span", "mono", ` ■ ${c.label}` );
    d.style.color = c.color; legend.appendChild(d); legend.appendChild(el("span", null, "  "));
  });
  box.appendChild(legend);
  return box;
}

let tipEl = null;
export function showTip(x, y, html) {
  if (!tipEl) tipEl = $("tooltip");
  if (x == null) { tipEl.hidden = true; return; }
  tipEl.innerHTML = html; tipEl.hidden = false;
  tipEl.style.left = Math.min(x + 12, innerWidth - 240) + "px";
  tipEl.style.top = (y + 14) + "px";
}
