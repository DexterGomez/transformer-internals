"""transformer-internals engine — one sentence, two architectures, opened up.

Left pane: bert-base-uncased (encoder, post-LN, MLM). Right pane: gpt2
(decoder, pre-LN, causal LM). Same geometry: 12 layers, 12 heads, 768 dims.

Every manual computation self-verifies against the model's own output
(attention rows: <1e-6; full-layer recompute: <1e-5) and the verify diffs
are returned in the payloads so the UI can show them as badges.
"""
from __future__ import annotations

import functools
import math
import threading

import numpy as np
import torch
import torch.nn.functional as F
from transformers import (AutoModel, AutoModelForCausalLM, AutoModelForMaskedLM,
                          AutoTokenizer)

MAX_TOKENS = 64
BERT_ID = "bert-base-uncased"
GPT2_ID = "openai-community/gpt2"

REGISTRY = {
    BERT_ID: {
        "label": "BERT-base · encoder", "family": "encoder", "layers": 12,
        "heads": 12, "dim": 768, "d_head": 64,
        "note": "Bidirectional attention · masked-LM · post-LN · "
                "token_type table · 512 positions · WordPiece",
    },
    GPT2_ID: {
        "label": "GPT-2 small · decoder", "family": "decoder", "layers": 12,
        "heads": 12, "dim": 768, "d_head": 64,
        "note": "Causal attention · next-token LM · pre-LN · no segment "
                "table · 1024 positions · byte-level BPE",
    },
}

_MODELS: dict[str, tuple] = {}
_LM_MODELS: dict[str, tuple] = {}
_LOAD_LOCK = threading.Lock()   # FastAPI threadpool: never load checkpoints in parallel


def get_model(model_id: str):
    with _LOAD_LOCK:
        if model_id not in _MODELS:
            tok = AutoTokenizer.from_pretrained(model_id)
            model = AutoModel.from_pretrained(
                model_id, attn_implementation="eager").eval()
            _MODELS[model_id] = (tok, model)
        return _MODELS[model_id]


def get_lm_model(model_id: str):
    with _LOAD_LOCK:
        if model_id not in _LM_MODELS:
            tok = AutoTokenizer.from_pretrained(model_id)
            model = (AutoModelForMaskedLM.from_pretrained(model_id)
                     if model_id == BERT_ID
                     else AutoModelForCausalLM.from_pretrained(model_id)).eval()
            _LM_MODELS[model_id] = (tok, model)
        return _LM_MODELS[model_id]


def meta_payload() -> dict:
    return {"models": [{"id": k, **v} for k, v in REGISTRY.items()],
            "default_text": "The animal didn't cross the street because it was too tired.",
            "compare": [
                ["attention", "bidirectional (every token sees every token)",
                 "causal (each token sees only the past)"],
                ["pretraining objective", "fill in 15% masked blanks (MLM) + NSP",
                 "predict every next token"],
                ["special tokens", "[CLS] … [SEP] (added automatically)", "none"],
                ["segment table (token_type)", "yes — 2×768, sentence A/B",
                 "absent — no sentence-pair objective"],
                ["positions", "learned, 512 rows (hard cap)", "learned, 1024 rows"],
                ["tokenizer", "WordPiece · 30,522 vocab · ## continuations",
                 "byte-level BPE · 50,257 vocab · ‘·’ marks a leading space"],
                ["LayerNorm placement", "post-LN (normalize after each residual add)",
                 "pre-LN (normalize before each sublayer)"],
                ["embedding output", "LayerNorm applied at the embedding stage",
                 "raw sum of token + position (no norm)"],
                ["native interface", 'fill the blank: "…too [MASK]."',
                 'continue the text: "…too" → ?'],
                ["parameters", "~110M", "~124M"],
                ["geometry", "12 layers · 12 heads · 768 dims", "same — 12·12·768"],
            ]}


def _encode(model_id: str, text: str):
    tok, model = get_model(model_id)
    enc = tok(text, return_tensors="pt")
    n = enc.input_ids.size(1)
    if n > MAX_TOKENS:
        raise ValueError(f"{n} tokens — keep it under {MAX_TOKENS}")
    return tok, model, enc, n


def _np(x, r=4):
    return np.round(x.detach().float().cpu().numpy(), r).tolist()


def _trunk(model):
    """GPT-2 handle that works pre/post the transformers-v5 flattening."""
    return getattr(model, "transformer", model)


def tokenize_payload(model_id: str, text: str) -> dict:
    tok, model, enc, n = _encode(model_id, text)
    ids = enc.input_ids[0].tolist()
    tokens = tok.convert_ids_to_tokens(ids)
    is_bert = model_id == BERT_ID
    special = [t in ("[CLS]", "[SEP]", "[MASK]") for t in tokens]
    subword = [t.startswith("##") for t in tokens]
    emb = (model.embeddings if is_bert else _trunk(model))
    tables = []
    if is_bert:
        for name, mod in [("word_embeddings", emb.word_embeddings),
                          ("position_embeddings", emb.position_embeddings),
                          ("token_type_embeddings", emb.token_type_embeddings)]:
            tables.append({"name": name, "shape": list(mod.weight.shape)})
    else:
        for name, mod in [("wte (token)", emb.wte), ("wpe (position)", emb.wpe)]:
            tables.append({"name": name, "shape": list(mod.weight.shape)})
    return {
        "model": model_id, "tokens": tokens, "input_ids": ids,
        "position_ids": list(range(n)),
        "token_type_ids": enc.token_type_ids[0].tolist() if is_bert else None,
        "special": special, "subword": subword, "n": n, "tables": tables,
    }


def _forward(model_id: str, text: str):
    tok, model, enc, n = _encode(model_id, text)
    with torch.no_grad():
        out = model(**enc, output_attentions=True, output_hidden_states=True)
    return tok, model, enc, out, n


def forward_payload(model_id: str, text: str) -> dict:
    tok, model, enc, out, n = _forward(model_id, text)
    tokens = tok.convert_ids_to_tokens(enc.input_ids[0])
    attn = torch.stack(out.attentions)[:, 0]                    # (L,H,N,N)
    is_bert = model_id == BERT_ID
    with torch.no_grad():
        if is_bert:
            e = model.embeddings
            ids, ttys = enc.input_ids, enc.token_type_ids
            pos = torch.arange(n)
            e_tok = e.word_embeddings(ids)[0]
            e_pos = e.position_embeddings(pos)
            e_typ = e.token_type_embeddings(ttys)[0]
            e_sum = e_tok + e_pos + e_typ
            e_out = e.LayerNorm(e_sum[None])[0]
            tok_type_ids = ttys[0].tolist()
        else:
            tr = _trunk(model)
            ids = enc.input_ids
            pos = torch.arange(n)
            e_tok = tr.wte(ids)[0]
            e_pos = tr.wpe(pos)
            e_typ = None
            e_sum = e_tok + e_pos
            e_out = e_sum
            tok_type_ids = None
    norms = lambda v: [round(float(x), 2) for x in v.norm(dim=1)] if v is not None else None
    hidden_norms = [[round(float(h[0][i].norm()), 2) for i in range(n)]
                    for h in out.hidden_states]
    return {
        "model": model_id, "tokens": tokens, "n": n,
        "token_type_ids": tok_type_ids,
        "emb_norms": {"token": norms(e_tok), "position": norms(e_pos),
                      "token_type": norms(e_typ), "sum": norms(e_sum),
                      "output": norms(e_out)},
        "emb_has_ln": is_bert,
        "num_layers": attn.shape[0], "num_heads": attn.shape[1],
        "attention": np.round(attn.numpy(), 4).tolist(),
        "hidden_norms": hidden_norms,
    }


def _qkv_heads(model, layer: int, x: torch.Tensor, is_bert: bool):
    """x: (N,768) input to the layer -> q,k,v each (N,H,d).

    Post-LN (BERT): attention consumes x directly. Pre-LN (GPT-2): attention
    consumes ln_1(x) — the normalized copy — while the residual keeps raw x.
    """
    n = x.size(0)
    if is_bert:
        sa = model.encoder.layer[layer].attention.self
        q = (x @ sa.query.weight.T + sa.query.bias).view(n, 12, 64)
        k = (x @ sa.key.weight.T + sa.key.bias).view(n, 12, 64)
        v = (x @ sa.value.weight.T + sa.value.bias).view(n, 12, 64)
    else:
        blk = _trunk(model).h[layer]
        qkv = blk.attn.c_attn(blk.ln_1(x))                        # (N, 2304)
        q, k, v = (t.view(n, 12, 64) for t in qkv.split(768, dim=1))
    return q, k, v


def projections_payload(model_id: str, text: str, layer: int, token: int) -> dict:
    tok, model, enc, out, n = _forward(model_id, text)
    tokens = tok.convert_ids_to_tokens(enc.input_ids[0])
    layer = int(max(0, min(layer, 11)))
    token = int(max(0, min(token, n - 1)))
    is_bert = model_id == BERT_ID
    x = out.hidden_states[layer][0].detach()                      # input to layer
    q, k, v = _qkv_heads(model, layer, x, is_bert)
    w_shapes = ([["W_query", 768, 768], ["W_key", 768, 768], ["W_value", 768, 768]]
                if is_bert else [["W_c_attn (Q,K,V fused)", 768, 2304]])
    return {
        "model": model_id, "tokens": tokens, "n": n, "layer": layer,
        "token": token, "token_str": tokens[token],
        "x_stripe": _np(x[token][::4]),                           # 192 dims
        "q_heads": _np(q[token]), "k_heads": _np(k[token]),
        "v_heads": _np(v[token]),                                 # each (12, 64)
        "w_shapes": w_shapes, "d_head": 64, "scale": 8.0,
    }


def row_payload(model_id: str, text: str, layer: int, head: int, query: int) -> dict:
    tok, model, enc, out, n = _forward(model_id, text)
    tokens = tok.convert_ids_to_tokens(enc.input_ids[0])
    layer = int(max(0, min(layer, 11)))
    head = int(max(0, min(head, 11)))
    query = int(max(0, min(query, n - 1)))
    is_bert = model_id == BERT_ID
    x = out.hidden_states[layer][0].detach()
    q, k, v = _qkv_heads(model, layer, x, is_bert)
    raw = (q[query, head] * k[:, head]).sum(-1) / 8.0             # scaled scores
    causal = None if is_bert else torch.triu(torch.ones(n, n, dtype=torch.bool), 1)
    row = (raw.masked_fill(causal[query], float("-inf"))
           if causal is not None else raw)
    w_manual = row.softmax(-1)
    w_model = torch.stack(out.attentions)[layer, 0, head, query].detach()
    diff = float((w_manual - w_model).abs().max())
    return {
        "model": model_id, "tokens": tokens, "n": n, "layer": layer,
        "head": head, "query": query, "query_str": tokens[query],
        "raw_scaled": _np(raw), "weights": _np(w_manual),
        "masked": causal[query].tolist() if causal is not None else [False] * n,
        "verify_diff": f"{diff:.1e}",
        "is_decoder": not is_bert,
    }


def assembly_payload(model_id: str, text: str, layer: int) -> dict:
    tok, model, enc, out, n = _forward(model_id, text)
    tokens = tok.convert_ids_to_tokens(enc.input_ids[0])
    layer = int(max(0, min(layer, 11)))
    is_bert = model_id == BERT_ID
    x = out.hidden_states[layer][0].detach()
    with torch.no_grad():
        if is_bert:
            sa, so = model.encoder.layer[layer].attention.self, \
                model.encoder.layer[layer].attention.output
            q, k, v = _qkv_heads(model, layer, x, True)
            w = (torch.einsum("ihd,jhd->hij", q, k) / 8.0).softmax(-1)
            ctx = torch.einsum("hij,jhd->ihd", w, v).reshape(n, 768)
            wo = so.dense(ctx)
            a = so.LayerNorm(x + wo)
            L = model.encoder.layer[layer]
            inter = F.gelu(L.intermediate.dense(a))
            h = L.output.LayerNorm(a + L.output.dense(inter))
            stages = [
                ("x (input to layer)", x, "(N, 768) — the residual stream"),
                ("per-head mix  ×12", ctx, "(N, 64) × 12 — concat → (N, 768)"),
                ("after W_O", wo, "(N, 768) — heads re-coordinated"),
                ("add residual + LayerNorm", a, "(N, 768) — post-LN"),
                ("FFN expand 768→3072 (GELU)", inter, "(N, 3072)"),
                ("project back + add + LN", h, "(N, 768) — layer output"),
            ]
            order = ["attention → +x → LayerNorm → FFN → + → LayerNorm"]
        else:
            b = _trunk(model).h[layer]
            x1 = b.ln_1(x)
            q, k, v = _qkv_heads(model, layer, x, False)
            w = (torch.einsum("ihd,jhd->hij", q, k) / 8.0)
            w = w.masked_fill(torch.triu(torch.ones(n, n, dtype=torch.bool), 1),
                              float("-inf")).softmax(-1)
            ctx = torch.einsum("hij,jhd->ihd", w, v).reshape(n, 768)
            wo = b.attn.c_proj(ctx)
            a = x + wo
            x2 = b.ln_2(a)
            inter = b.mlp.act(b.mlp.c_fc(x2))
            h = a + b.mlp.c_proj(inter)
            stages = [
                ("LayerNorm first (pre-LN)", x1, "(N, 768) — normalized copy"),
                ("per-head causal mix ×12", ctx, "(N, 64) × 12 — concat → (N, 768)"),
                ("after c_proj (W_O)", wo, "(N, 768) — heads re-coordinated"),
                ("add residual", a, "(N, 768) — x was never normalized"),
                ("LayerNorm → FFN 768→3072", inter, "(N, 3072)"),
                ("project back + add", h, "(N, 768) — layer output"),
            ]
            order = ["LayerNorm → attention → +x → LayerNorm → FFN → +x"]
    target = out.hidden_states[layer + 1][0].detach()
    diff = float((h - target).abs().max())
    stage_list = [{"name": nm,
                   "mean_norm": round(float(t.norm(dim=1).mean()), 2),
                   "shape": sh} for nm, t, sh in stages]
    return {
        "model": model_id, "tokens": tokens, "n": n, "layer": layer,
        "order": order, "stages": stage_list,
        "verify_diff": f"{diff:.1e}",
        "params": {"attention": "2.36M", "ffn": "4.72M",
                   "note": "two-thirds of every layer is the FFN"},
    }


def task_payload(model_id: str, text: str, temperature: float = 1.0,
                 top_k: int = 50, n_new: int = 8) -> dict:
    tok, model = get_lm_model(model_id)
    with torch.no_grad():
        enc = tok(text, return_tensors="pt")
        n = enc.input_ids.size(1)
        if n > MAX_TOKENS:
            raise ValueError(f"{n} tokens — keep it under {MAX_TOKENS}")
        out = model(**enc)
        if model_id == BERT_ID:
            mask_id = tok.mask_token_id
            positions = (enc.input_ids[0] == mask_id).nonzero().flatten().tolist()
            if not positions:
                return {"model": model_id, "error":
                        "Add a [MASK] token to the text to probe the encoder's "
                        "native task (fill the blank)."}
            probs = out.logits[0, positions[0]].softmax(-1)
            top = torch.topk(probs, 8)
            preds = [{"token": tok.convert_ids_to_tokens(int(i)),
                      "p": round(float(p), 4)}
                     for i, p in zip(top.indices, top.values)]
            return {"model": model_id, "kind": "mask",
                    "masked_positions": positions, "predictions": preds}
        probs = out.logits[0, -1].softmax(-1)
        top = torch.topk(probs, 8)
        preds = [{"token": tok.convert_ids_to_tokens(int(i)),
                  "p": round(float(p), 4)}
                 for i, p in zip(top.indices, top.values)]
        gen_kwargs = {"max_new_tokens": int(n_new), "pad_token_id": tok.eos_token_id}
        if temperature > 0.01:
            gen_kwargs.update(do_sample=True, temperature=float(temperature),
                              top_k=int(top_k))
        else:
            gen_kwargs["do_sample"] = False      # greedy: always the top token
        gen = model.generate(**enc, **gen_kwargs)
        new_tokens = tok.convert_ids_to_tokens(gen[0][n:].tolist())
        return {"model": model_id, "kind": "next",
                "predictions": preds, "continuation": tok.decode(gen[0][n:]),
                "continuation_tokens": new_tokens,
                "temperature": temperature, "top_k": top_k}
