# transformer-internals — encoder vs decoder, one layer at a time

An interactive tool for opening up a Transformer. One sentence, two models
side by side — `bert-base-uncased` (encoder) and `gpt2` (decoder, small) —
with every internal stage opened and **recomputed by hand from the
weights**, not just read off the forward pass. Same geometry on both sides
(12 layers · 12 heads · 768 dims), so every visible difference is
architecture, not size.

FastAPI + vanilla JS, no build step, no CDNs. Once the two checkpoints are
downloaded it runs fully offline.

## Install and run

Requires Python 3.11+. On first use each model (~1 GB combined) downloads
from Hugging Face and is cached under `~/.cache/huggingface`.

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python app.py        # → http://127.0.0.1:8310
```

The first forward pass per model takes ~20 s (checkpoint loads into
memory); after that, payloads are cached per model+text and feel instant.
The server binds `127.0.0.1:8310` — change it at the bottom of `app.py`.

## The five tabs (and the story they tell)

1. **Inputs** — the three id arrays and their tables. BERT sums
   word + position + token_type and applies LayerNorm; GPT-2 sums
   token + position and ships it raw. The decoder's missing segment table
   is the objective fingerprint beat: no sentence pairs, no provenance tag.
2. **Projections** — a selected token's hidden state, then Q/K/V as a
   (12, 64) grid: one row per head. The `view` made literal — zero
   parameters, head *h* = dims 64h…64h+63. GPT-2's fused c_attn
   (768→2304) vs BERT's three separate matrices is visible in the cards.
3. **Attention** — the (N, N) grid per layer/head with hover values, the
   query row sorted as bars with raw q·k/8 annotations, and the causal
   mask shading the decoder's upper triangle (−∞ before softmax). Default
   sliders (L3·H10, token *it*) land on a row that puts 84.7% on
   *animal*. The badge on each pane reports the manual-vs-model diff.
   The same head flips to an **arc diagram** — the sentence drawn twice,
   one arrow per query→key edge, thickness = weight — where the encoder's
   fans land on both sides and the decoder's only on the past: the mask
   as missing arrows.
4. **The layer** — the stage flow recomputed from weights: mix → W_O →
   residual (+LN order differs!) → FFN 768→3072→768. BERT is post-LN,
   GPT-2 pre-LN; the badge confirms the by-hand layer equals the model's
   hidden state (≤1e-6).
5. **Native task** — same weights, different interface: BERT fills a
   `[MASK]` (top predictions with honest, uncertain probabilities); GPT-2
   continues the text with live temperature / top-k / length — the
   decoding loop, one token at a time.

## Beats to narrate

- Tab 1 → "read the embedding tables and you can infer the pretraining
  objective." Tab 3 → "the leak argument: if position 5 sees position 6,
  next-token training is cheating — so the mask is a *necessity*."
- Tab 3 mask toggle off/on is the one-line diff between the branches;
  the same L3·H10-ish row on GPT-2 shows budget crammed into the past.
- Tab 4 → "attention moves information, the FFN thinks — and two-thirds
  of every layer is the FFN."
- Tab 5 closes the arc: encoder repairs, decoder continues; generation
  = the decoding loop you just steered with sliders.

## Engine self-verification

`internals.py` recomputes attention rows and full layers from raw weights;
payloads carry the max-abs diff vs the model's own output (attention rows
~1e-8, full layers ≤1e-6 for BERT, 0 for GPT-2 — GPT-2's path calls the
same submodules, so it matches bitwise). If a badge ever shows a large
diff, something regressed — trust the badge, not the pretty colors.

## Project layout

- `app.py` — FastAPI server (seven JSON endpoints, static file serving)
- `internals.py` — the engine: tokenization, manual recompute, verification
- `static/` — the UI (vanilla JS + CSS, no dependencies)
