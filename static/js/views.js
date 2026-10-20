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
  if (x == null || x === 0) { tipEl.hidden = true; return; }
  tipEl.innerHTML = html; tipEl.hidden = false;
  tipEl.style.left = Math.min(x + 12, innerWidth - 240) + "px";
  tipEl.style.top = (y + 14) + "px";
}

/* -------- paper-style arc view of one attention head --------
   The sentence twice: keys on top, queries on the bottom. One arrow
   per (query → key) edge, weight encoded as thickness + opacity. On a
   causal model the future simply receives no arrows — the mask made
   visible as absence. Hover a token to isolate its fan. */
const FAM_HEX = { bert: "#B0731E", gpt2: "#3E6B99" };
let arcUid = 0;

export function arcDiagram(host, tokens, weights, fam, { sel = -1, onPick = null } = {}) {
  const n = tokens.length;
  const hex = FAM_HEX[fam] ?? "#3E6B99";
  const NS = "http://www.w3.org/2000/svg";
  const boxH = 24, topY = 38, botY = 138, H = 192, gap = 7;
  const uid = "ah" + (++arcUid);

  const widths = tokens.map(t => Math.max(26, t.length * 6.7 + 11));
  const total = widths.reduce((a, b) => a + b, 0) + gap * (n - 1);
  const avail = Math.max(host.clientWidth || 0, 320) - 2;
  const gaps = (n > 1 && total + 16 < avail) ? gap + (avail - 16 - total) / (n - 1) : gap;
  const xs = [], cx = [];
  let x = 8;
  for (let j = 0; j < n; j++) { xs.push(x); cx.push(x + widths[j] / 2); x += widths[j] + gaps; }
  const W = Math.max(x - gaps + 8, avail);

  const svg = document.createElementNS(NS, "svg");
  svg.setAttribute("class", "arcsvg");
  svg.setAttribute("width", W); svg.setAttribute("height", H);
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.style.display = "block"; svg.style.background = "#ffffff";
  host.appendChild(svg);
  const mk = (t, attrs, parent = svg) => {
    const e = document.createElementNS(NS, t);
    for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
    parent.appendChild(e); return e;
  };

  const defs = mk("defs", {});
  const marker = mk("marker", { id: uid, markerUnits: "userSpaceOnUse",
    markerWidth: 7, markerHeight: 6, refX: 6.4, refY: 3, orient: "auto" }, defs);
  mk("path", { d: "M0,0 L7,3 L0,6 Z", fill: hex }, marker);

  const keepPer = n <= 28 ? n : 8;                      // cap the hairball on long inputs
  const edges = [];
  for (let i = 0; i < n; i++) {
    const order = Array.from({ length: n }, (_, j) => j)
      .sort((a, b) => weights[i][b] - weights[i][a]).slice(0, keepPer);
    for (const j of order) {
      if (weights[i][j] >= 0.004) edges.push({ q: i, k: j, w: weights[i][j] });
    }
  }
  edges.sort((a, b) => a.w - b.w);                      // weak first, strong on top

  const gArc = mk("g", {});
  const drawn = [];
  for (const e of edges) {
    const x1 = cx[e.q], x2 = cx[e.k];
    const d = `M ${x1} ${botY} C ${x1} ${botY - 34}, ${x2} ${topY + boxH + 34}, ${x2} ${topY + boxH}`;
    const op = 0.10 + 0.90 * Math.min(1, e.w * 1.7);
    const sw = 0.8 + 6.5 * Math.pow(e.w, 0.6);
    const p = mk("path", { d, fill: "none", stroke: hex, "stroke-width": sw,
      opacity: op, "marker-end": `url(#${uid})` }, gArc);
    drawn.push({ ...e, p, op });
    const hit = mk("path", { d, fill: "none", stroke: "transparent",
      "stroke-width": 10, "pointer-events": "stroke", cursor: "crosshair" }, gArc);
    hit.addEventListener("mousemove", ev => showTip(ev.clientX, ev.clientY,
      `${tokens[e.q]} → ${tokens[e.k]} · <span class="mono">${(e.w * 100).toFixed(1)}%</span>`));
    hit.addEventListener("mouseleave", () => showTip(0, 0, null));
  }

  const boxes = tokens.map(() => ({}));
  const mkBox = (j, y, role) => {
    const g = mk("g", { cursor: "pointer" });
    const special = tokens[j].startsWith("[");
    const r = mk("rect", { x: xs[j], y, width: widths[j], height: boxH, rx: 5,
      fill: special ? "#f6f7f9" : "#ffffff", stroke: "#d9dde5",
      ...(special ? { "stroke-dasharray": "3 2" } : {}) }, g);
    const t = mk("text", { x: cx[j], y: y + boxH / 2 + 3.5, "text-anchor": "middle",
      "font-family": "ui-monospace, Menlo, monospace", "font-size": "11",
      fill: "#1f2430" }, g);
    t.textContent = tokens[j];
    const focusIn = () => { focus = { dir: role, idx: j }; apply(); };
    const focusOut = () => { focus = null; apply(); };
    g.addEventListener("mouseenter", focusIn);
    g.addEventListener("mouseleave", focusOut);
    g.addEventListener("click", () => onPick?.(j));
    return { r };
  };

  const labelG = mk("g", {});
  let focus = null;
  for (let j = 0; j < n; j++) {
    boxes[j].top = mkBox(j, topY, "k");
    boxes[j].bot = mkBox(j, botY, "q");
  }

  function apply() {
    labelG.innerHTML = "";
    for (const e of drawn) {
      let o;
      if (focus) {
        const inFan = focus.dir === "q" ? e.q === focus.idx : e.k === focus.idx;
        o = inFan ? Math.max(e.op, 0.95) : 0.05;
      } else if (sel >= 0) {
        o = e.q === sel ? Math.min(1, e.op + 0.3) : e.op * 0.4;
      } else o = e.op;
      e.p.setAttribute("opacity", o);
    }
    for (let j = 0; j < n; j++) {
      const selQ = j === sel && !focus;
      const hovQ = focus?.dir === "q" && focus.idx === j;
      const hovK = focus?.dir === "k" && focus.idx === j;
      boxes[j].bot.r.setAttribute("stroke", (selQ || hovQ) ? hex : "#d9dde5");
      boxes[j].bot.r.setAttribute("stroke-width", (selQ || hovQ) ? 2 : 1);
      boxes[j].top.r.setAttribute("stroke", hovK ? hex : "#d9dde5");
      boxes[j].top.r.setAttribute("stroke-width", hovK ? 2 : 1);
      boxes[j].top.r.setAttribute("fill", hovK ? "#f2ece3" : (tokens[j].startsWith("[") ? "#f6f7f9" : "#ffffff"));
    }
    if (focus) {
      const fan = drawn.filter(e => focus.dir === "q" ? e.q === focus.idx : e.k === focus.idx)
        .sort((a, b) => b.w - a.w).slice(0, 3);
      for (const e of fan) {
        const pct = (e.w * 100).toFixed(1) + "%" + (e.q === e.k ? " self" : "");
        const attrs = { "text-anchor": "middle", "font-size": "10", fill: hex, "font-weight": "600" };
        const t = focus.dir === "q"
          ? mk("text", { ...attrs, x: cx[e.k], y: topY - 8 }, labelG)
          : mk("text", { ...attrs, x: cx[e.q], y: botY + boxH + 14 }, labelG);
        t.textContent = pct;
      }
    }
  }
  svg.addEventListener("mouseleave", () => { focus = null; apply(); });
  apply();

  return { svg, truncated: keepPer < n };
}
