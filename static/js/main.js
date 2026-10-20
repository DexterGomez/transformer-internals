/* transformer-internals — dual-pane controller */
import { $, el, drawMatrix, drawStripe, chipRow, stackedNorms, showTip, barColor, disp } from "./views.js";

const MODELS = [
  { id: "bert-base-uncased", fam: "bert", cls: "bert", name: "BERT-base · encoder" },
  { id: "openai-community/gpt2", fam: "gpt2", cls: "gpt2", name: "GPT-2 small · decoder" },
];

const state = {
  text: "", meta: null, tab: "inputs",
  tokenize: {}, forward: {},
  projLayer: 3, projQKV: "q", projToken: {},
  attLayer: 3, attHead: 10, attQuery: {}, maskOn: true,
  lyrLayer: 3,
  task: { maskText: "", temperature: 1.0, topK: 50, nNew: 8, results: {} },
};

/* ---------------- fetch helpers ---------------- */
async function post(path, body) {
  const r = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body) });
  return r.json();
}
const setStatus = (t) => ($("status").textContent = t);
const famOf = (fam) => MODELS.find(m => m.fam === fam);

function errBox(msg) {
  const d = el("div", "error", msg);
  return d;
}

/* ---------------- boot ---------------- */
async function boot() {
  const meta = await (await fetch("/api/meta")).json();
  state.meta = meta;
  $("text-in").value = meta.default_text;
  const tbl = $("compare-table");
  const head = el("tr");
  ["feature", "BERT (encoder)", "GPT-2 (decoder)"].forEach(h => head.appendChild(el("th", null, h)));
  tbl.appendChild(head);
  meta.compare.forEach(row => {
    const tr = el("tr");
    row.forEach(c => tr.appendChild(el("td", null, c)));
    tbl.appendChild(tr);
  });
  $("compare-toggle").onclick = () => {
    const t = $("compare-table");
    t.hidden = !t.hidden;
    $("compare-toggle").textContent = (t.hidden ? "▸" : "▾") + " architecture comparison";
  };
  $("run-btn").onclick = run;
  $("text-in").addEventListener("keydown", e => { if (e.key === "Enter") run(); });
  document.querySelectorAll(".tab").forEach(b => b.onclick = () => {
    document.querySelectorAll(".tab").forEach(x => x.classList.remove("active"));
    b.classList.add("active");
    state.tab = b.dataset.tab;
    document.querySelectorAll(".tab-panel").forEach(p => p.hidden = p.id !== "tab-" + state.tab);
    renderTab();
  });
  $("proj-layer").oninput = e => { state.projLayer = +e.target.value; $("proj-layer-label").textContent = "L" + (state.projLayer + 1); renderProjections(); };
  $("proj-qkv").onchange = e => { state.projQKV = e.target.value; renderProjections(); };
  $("att-layer").oninput = e => { state.attLayer = +e.target.value; $("att-layer-label").textContent = "L" + (state.attLayer + 1); renderAttention(); };
  $("att-head").oninput = e => { state.attHead = +e.target.value; $("att-head-label").textContent = "H" + (state.attHead + 1); renderAttention(); };
  $("att-mask").onchange = e => { state.maskOn = e.target.checked; renderAttention(); };
  $("lyr-layer").oninput = e => { state.lyrLayer = +e.target.value; $("lyr-layer-label").textContent = "L" + (state.lyrLayer + 1); renderLayer(); };
  await run();
}

async function run() {
  state.text = $("text-in").value.trim();
  if (!state.text) return;
  setStatus("running both models…");
  state.tokenize = {}; state.forward = {}; state.task.results = {};
  for (const m of MODELS) {
    state.tokenize[m.fam] = await post("/api/tokenize", { model: m.id, text: state.text });
    state.forward[m.fam] = await post("/api/forward", { model: m.id, text: state.text });
  }
  for (const m of MODELS) {
    const t = state.tokenize[m.fam];
    if (t.error) continue;
    const want = m.fam === "bert" ? "it" : "Ġit";
    const i = t.tokens.indexOf(want);
    state.attQuery[m.fam] = i >= 0 ? i : Math.floor(t.n / 2);
    state.projToken[m.fam] = state.attQuery[m.fam];
  }
  state.task.maskText = state.text.replace(/too\s+\w+\./, "too [MASK].");
  state.task.gptPrompt = state.text;
  setStatus(`${state.tokenize.bert?.n ?? "?"} tokens (BERT) · ${state.tokenize.gpt2?.n ?? "?"} tokens (GPT-2) — ready`);
  renderTab();
}

function renderTab() {
  ({ inputs: renderInputs, projections: renderProjections, attention: renderAttention,
     layer: renderLayer, task: renderTask })[state.tab]();
}

const PANE_PREFIX = { inputs: "inputs", projections: "proj", attention: "att",
                      layer: "lyr", task: "task" };
function pane(fam) { return $(PANE_PREFIX[state.tab] + "-" + fam); }

function head3(fam, extra) {
  const m = famOf(fam);
  const h = el("h3", null, m.name);
  if (extra) h.appendChild(el("span", "badge", extra));
  return h;
}

/* ---------------- tab 1: inputs ---------------- */
function renderInputs() {
  for (const m of MODELS) {
    const p = pane(m.fam);
    p.innerHTML = "";
    const t = state.tokenize[m.fam], f = state.forward[m.fam];
    if (t.error || f.error) { p.appendChild(errBox(t.error || f.error)); continue; }
    p.appendChild(head3(m.fam));
    p.appendChild(chipRow(t.tokens.map(disp), { special: t.special, sub: t.subword }));
    const cols = el("div", "idcols");
    [["input_ids", t.input_ids], ["position_ids", t.position_ids],
     ["token_type_ids", t.token_type_ids]].forEach(([lbl, vals]) => {
      const c = el("div", "idcol");
      c.appendChild(el("div", "lbl", lbl));
      const v = el("div", "vals mono");
      (vals ?? ["—"]).forEach(x => v.appendChild(el("span", null, x)));
      c.appendChild(v);
      cols.appendChild(c);
    });
    p.appendChild(cols);
    const cards = el("div", "cards");
    t.tables.forEach(tb => {
      const c = el("div", "card");
      c.appendChild(el("b", null, tb.name));
      c.appendChild(el("div", "mono", `${tb.shape[0]} × ${tb.shape[1]}`));
      cards.appendChild(c);
    });
    if (m.fam === "gpt2") {
      const c = el("div", "card");
      c.appendChild(el("b", null, "token_type table"));
      c.appendChild(el("div", "mono", "absent — no sentence-pair objective"));
      cards.appendChild(c);
    }
    p.appendChild(cards);
    const formula = el("div", "formula");
    formula.innerHTML = m.fam === "bert"
      ? "E = <b>word</b><span class='op'>[token]</span> + <b>position</b><span class='op'>[slot]</span> + <b>token_type</b><span class='op'>[A/B]</span> → LayerNorm"
      : "E = <b>wte</b><span class='op'>[token]</span> + <b>wpe</b><span class='op'>[slot]</span> <span class='op'>(no norm at the embedding stage)</span>";
    p.appendChild(formula);
    const en = f.emb_norms;
    const comps = [
      { label: "token", color: barColor(m.fam), norms: en.token },
      { label: "position", color: "#7F8C9B", norms: en.position },
    ];
    if (en.token_type) comps.push({ label: "token_type", color: "#B279A2", norms: en.token_type });
    p.appendChild(stackedNorms(t.tokens, comps, m.fam));
    p.appendChild(el("div", "hint", m.fam === "bert"
      ? `BERT's embedding LayerNorm rescales every token (output norms ≈ ${en.output[0]}), GPT-2 ships the raw sum upstream.`
      : `GPT-2's sum goes in unnormalized (output norms ≈ ${en.output[0]}) — the first LayerNorm appears inside layer 1 (pre-LN).`));
  }
}

/* ---------------- tab 2: projections ---------------- */
async function renderProjections() {
  for (const m of MODELS) {
    const p = pane(m.fam);
    p.innerHTML = "";
    const tk = state.tokenize[m.fam];
    if (tk.error) { p.appendChild(errBox(tk.error)); continue; }
    p.appendChild(head3(m.fam));
    p.appendChild(chipRow(tk.tokens.map(disp), { special: tk.special, sel: state.projToken[m.fam],
      onPick: (i) => { state.projToken[m.fam] = i; renderProjections(); } }));
    const r = await post("/api/projections", { model: m.id, text: state.text,
      layer: state.projLayer, token: state.projToken[m.fam] });
    if (r.error) { p.appendChild(errBox(r.error)); continue; }
    p.appendChild(el("div", "hint", `layer ${r.layer + 1} · token “${disp(r.token_str)}” — its input vector, then the projection`));
    let b = el("div", "block");
    b.appendChild(el("h4", null, `x — hidden state (every 4th dim shown)`));
    let cv = el("canvas");
    b.appendChild(el("div", "scroll")).appendChild(cv);
    p.appendChild(b);
    drawStripe(cv, r.x_stripe, { cell: 7 });
    b = el("div", "block");
    const which = state.projQKV;
    b.appendChild(el("h4", null, `${which.toUpperCase()} viewed as (12, 64) — one row per head, zero math`));
    cv = el("canvas");
    b.appendChild(el("div", "scroll")).appendChild(cv);
    p.appendChild(b);
    drawStripe(cv, r[which + "_heads"], { cell: 11, rowLabels: Array.from({ length: 12 }, (_, h) => "H" + h) });
    const cards = el("div", "cards");
    r.w_shapes.forEach(([nm, a, bb]) => {
      const c = el("div", "card");
      c.appendChild(el("b", null, nm));
      c.appendChild(el("div", "mono", `${a} × ${bb} (learned)`));
      cards.appendChild(c);
    });
    const c3 = el("div", "card");
    c3.appendChild(el("b", null, "view (N,768)→(N,12,64)"));
    c3.appendChild(el("div", "mono", "no parameters — head h = dims 64h…64h+63"));
    cards.appendChild(c3);
    p.appendChild(cards);
    p.appendChild(el("div", "hint", "The linear map is the projection; the view is bookkeeping. ÷√64 = 8 happens later, on the scores."));
  }
}

/* ---------------- tab 3: attention ---------------- */
async function renderAttention() {
  for (const m of MODELS) {
    const p = pane(m.fam);
    p.innerHTML = "";
    const tk = state.tokenize[m.fam], fw = state.forward[m.fam];
    if (tk.error || fw.error) { p.appendChild(errBox(tk.error || fw.error)); continue; }
    const L = state.attLayer, H = state.attHead, q = state.attQuery[m.fam];
    p.appendChild(head3(m.fam));
    p.appendChild(chipRow(tk.tokens.map(disp), { special: tk.special, sel: q,
      onPick: (i) => { state.attQuery[m.fam] = i; renderAttention(); } }));
    const b = el("div", "block");
    b.appendChild(el("h4", null, `L${L + 1} · H${H + 1} — the (N, N) grid`));
    const cv = el("canvas");
    b.appendChild(el("div", "scroll")).appendChild(cv);
    p.appendChild(b);
    const n = tk.n;
    let shade = null;
    if (m.fam === "gpt2" && state.maskOn)
      shade = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => j > i));
    drawMatrix(cv, fw.attention[L][H], { rowLabels: tk.tokens.map(disp), colLabels: tk.tokens.map(disp), shadeMask: shade,
      cellW: 30, cellH: 22,
      onHover: (x, y, ri, ci, v, blocked) => showTip(x, y,
        `${ri} → ${ci}: <span class="mono">${(v * 100).toFixed(1)}%</span>${blocked ? " (−∞ before softmax)" : ""}`) });
    const r = await post("/api/row", { model: m.id, text: state.text, layer: L, head: H, query: q });
    if (r.error) { p.appendChild(errBox(r.error)); continue; }
    p.querySelector("h3").appendChild(
      el("span", "badge", `manual ≡ model (diff ${r.verify_diff})`));
    const rb = el("div", "block rowbars");
    rb.appendChild(el("h4", null, `row of “${disp(r.query_str)}” — sorted`));
    const idx = Array.from({ length: n }, (_, i) => i).sort((a, b2) => r.weights[b2] - r.weights[a]);
    const top = idx.slice(0, 12);
    top.forEach((j, rank) => {
      const rowEl = el("div", "rowbar");
      rowEl.appendChild(el("div", "tok mono", disp(r.tokens[j])));
      const bar = el("div", "bar");
      const fill = el("i");
      fill.style.width = (r.weights[j] * 100) + "%";
      fill.style.background = j === q ? "#E45756" : barColor(m.fam);
      bar.appendChild(fill);
      rowEl.appendChild(bar);
      rowEl.appendChild(el("div", "pc mono", (r.weights[j] * 100).toFixed(1) + "%"));
      rowEl.appendChild(el("div", "raw mono", rank < 4 ? `q·k/8 = ${r.raw_scaled[j].toFixed(2)}` :
        (r.masked[j] ? "masked −∞" : "")));
      rb.appendChild(rowEl);
    });
    p.appendChild(rb);
    p.appendChild(el("div", "hint", m.fam === "bert"
      ? "Bidirectional: the row spans every token. Sliders L4·H11 put 84.7% of the row on animal."
      : "Causal: cells above the diagonal are −∞ before softmax — the query row cannot spend budget on the future."));
  }
}

/* ---------------- tab 4: the layer ---------------- */
async function renderLayer() {
  for (const m of MODELS) {
    const p = pane(m.fam);
    p.innerHTML = "";
    const tk = state.tokenize[m.fam];
    if (tk.error) { p.appendChild(errBox(tk.error)); continue; }
    const r = await post("/api/assembly", { model: m.id, text: state.text, layer: state.lyrLayer });
    if (r.error) { p.appendChild(errBox(r.error)); continue; }
    p.appendChild(head3(m.fam, `recomputed by hand ≡ model (diff ${r.verify_diff})`));
    p.appendChild(el("div", "hint", r.order[0]));
    const maxN = Math.max(...r.stages.map(s => s.mean_norm), 0.01);
    r.stages.forEach((s, i) => {
      if (i) p.appendChild(el("div", "flowarrow", "↓"));
      const box = el("div", "flowbox");
      const nm = el("div", "nm"); nm.appendChild(el("b", null, s.name));
      box.appendChild(nm);
      box.appendChild(el("span", "sh mono", s.shape.split("—")[0].trim()));
      const nl = el("div", "normline");
      const bar = el("i"); bar.style.width = (s.mean_norm / maxN * 100) + "%";
      bar.style.background = barColor(m.fam);
      nl.appendChild(bar);
      box.appendChild(nl);
      box.appendChild(el("div", "hint", `mean ‖·‖ = ${s.mean_norm}`));
      p.appendChild(box);
    });
    const cards = el("div", "cards");
    [["attention", r.params.attention], ["FFN", r.params.ffn]].forEach(([nm, v]) => {
      const c = el("div", "card");
      c.appendChild(el("b", null, nm + " params"));
      c.appendChild(el("div", "mono", v));
      cards.appendChild(c);
    });
    p.appendChild(cards);
    p.appendChild(el("div", "hint", r.params.note + ". Every number above was recomputed from the weights — not read from the model's forward cache."));
  }
}

/* ---------------- tab 5: native task ---------------- */
function renderTask() {
  renderTaskBert();
  renderTaskGpt();
}

async function runTask(fam) {
  const m = famOf(fam);
  const body = fam === "bert"
    ? { model: m.id, text: state.task.maskText }
    : { model: m.id, text: state.task.gptPrompt || state.text,
        temperature: +state.task.temperature, top_k: +state.task.topK, n_new: +state.task.nNew };
  const r = await post("/api/task", body);
  state.task.results[fam] = r;
  if (fam === "bert") renderTaskBert(); else renderTaskGpt();
}

function predList(container, preds, fam) {
  const max = Math.max(...preds.map(x => x.p), 1e-6);
  preds.forEach(pr => {
    const rowEl = el("div", "predrow");
    rowEl.appendChild(el("div", "tok mono", disp(pr.token)));
    const bar = el("div", "bar");
    const fill = el("i");
    fill.style.width = (pr.p / max * 100) + "%";
    fill.style.background = barColor(fam);
    bar.appendChild(fill);
    rowEl.appendChild(bar);
    rowEl.appendChild(el("div", "pc mono", (pr.p * 100).toFixed(1) + "%"));
    container.appendChild(rowEl);
  });
}

function renderTaskBert() {
  const p = pane("bert");
  p.innerHTML = "";
  p.appendChild(head3("bert"));
  p.appendChild(el("div", "hint", "the encoder's interface: mask a word, ask for repairs"));
  const input = el("input");
  input.value = state.task.maskText || state.text;
  input.className = "mono"; input.style.cssText = "width:100%;background:var(--panel2);color:var(--text);border:1px solid var(--line);border-radius:8px;padding:8px 10px;font-size:13px";
  input.oninput = () => state.task.maskText = input.value;
  p.appendChild(input);
  const btn = el("button", "btn", "fill the blank");
  btn.style.marginTop = "8px";
  btn.onclick = () => runTask("bert");
  p.appendChild(btn);
  const r = state.task.results.bert;
  if (r?.error) p.appendChild(errBox(r.error));
  else if (r) {
    const b = el("div", "block");
    b.appendChild(el("h4", null, `top predictions for [MASK] at position ${r.masked_positions[0]}`));
    predList(b, r.predictions, "bert");
    p.appendChild(b);
    p.appendChild(el("div", "hint", "Honest probabilities: usually no certainty — the blank has more than one plausible filler."));
  }
}

function renderTaskGpt() {
  const p = pane("gpt2");
  p.innerHTML = "";
  p.appendChild(head3("gpt2"));
  p.appendChild(el("div", "hint", "the decoder's interface: condition on the text, predict the next token — temperature 0 = greedy"));
  const pv = el("input");
  pv.value = state.task.gptPrompt || state.text;
  pv.className = "mono";
  pv.style.cssText = "width:100%;background:var(--panel2);color:var(--text);border:1px solid var(--line);border-radius:8px;padding:8px 10px;font-size:13px";
  pv.placeholder = "prompt to continue";
  pv.addEventListener("input", () => state.task.gptPrompt = pv.value);
  p.appendChild(pv);
  p.appendChild(el("div", "hint", "↑ the prompt — edit here (or the top bar) and press continue"));
  const row = el("div", "controls-row");
  const mk = (lbl, val, min, max, step, cb) => {
    const l = el("label", "control", lbl + " ");
    const i = el("input"); i.type = "range"; i.min = min; i.max = max; i.step = step; i.value = val;
    const s = el("span", "mono", val);
    i.oninput = () => { s.textContent = i.value; cb(+i.value); };
    l.appendChild(i); l.appendChild(s);
    return l;
  };
  row.appendChild(mk("temperature", state.task.temperature, 0, 2, 0.1, v => state.task.temperature = v));
  row.appendChild(mk("top-k", state.task.topK, 1, 100, 1, v => state.task.topK = v));
  row.appendChild(mk("tokens", state.task.nNew, 4, 16, 1, v => state.task.nNew = v));
  p.appendChild(row);
  const btn = el("button", "btn", "continue the text");
  btn.onclick = () => runTask("gpt2");
  p.appendChild(btn);
  const r = state.task.results.gpt2;
  if (r?.error) p.appendChild(errBox(r.error));
  else if (r) {
    const b = el("div", "block");
    b.appendChild(el("h4", null, "next-token distribution (last position)"));
    predList(b, r.predictions, "gpt2");
    p.appendChild(b);
    const cont = el("div", "continuation");
    cont.appendChild(el("div", "hint", `continuation (T=${r.temperature}, k=${r.top_k}) — line breaks are real ↵ tokens:`));
    const body = el("div");
    body.style.whiteSpace = "pre-wrap";
    body.textContent = "…" + r.continuation;
    cont.appendChild(body);
    p.appendChild(cont);
    const chips = el("div", "chips");
    r.continuation_tokens.forEach(t => {
      const c = el("span", "chip", disp(t));
      c.style.cursor = "default";
      chips.appendChild(c);
    });
    p.appendChild(chips);
    p.appendChild(el("div", "hint", "One token at a time, each output fed back — this loop is decoding. After a period, GPT-2's top pick is a paragraph break (↵, ~22% in WebText) — trained on web pages, it ends stories. A 124M model from 2019: locally fluent, globally lost."));
  }
}

boot();
