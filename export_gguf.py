#!/usr/bin/env python3
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chat import EOD, GPT, find_model, read_ckpt

SPECIAL = ["<|end|>", "<|eod|>", "<|user|>", "<|bot|>"]
TEMPLATE = (
    "{% for m in messages %}{% if m['role'] == 'user' %}<|user|>{{ m['content'] }}<|end|><|bot|>"
    "{% else %}{{ m['content'] }}<|end|>{% endif %}{% endfor %}"
)


def permute(w, n_head):
    return (
        w.reshape(n_head, 2, w.shape[0] // n_head // 2, *w.shape[1:])
        .transpose(1, 2)
        .reshape(w.shape)
    )


def load_vocab(path, V, gguf):
    tj = json.load(open(path, encoding="utf-8"))
    vocab = tj["model"]["vocab"]
    if max(vocab.values()) >= V:
        raise SystemExit(f"tokenizer has ids up to {max(vocab.values())} but the model has {V} embeddings")
    tokens = [None] * V
    for t, i in vocab.items():
        tokens[i] = t
    if tokens[:4] != SPECIAL:
        raise SystemExit(f"expected the first four tokens to be {SPECIAL}, got {tokens[:4]}")
    types, unused = [], 0
    for i, t in enumerate(tokens):
        if t is None:
            tokens[i] = f"<unused_{i}>"
            types.append(gguf.TokenType.UNUSED)
            unused += 1
        elif i < 4:
            types.append(gguf.TokenType.CONTROL)
        else:
            types.append(gguf.TokenType.NORMAL)
    merges = [m if isinstance(m, str) else " ".join(m) for m in tj["model"]["merges"]]
    return tokens, types, merges, unused


def convert(args):
    import gguf

    path = find_model(args.model)
    ck = read_ckpt(path)
    cfg, sd = ck["cfg"], ck["model"]
    d, L, H, T, V = cfg["d"], cfg["L"], cfg["H"], cfg["T"], cfg["V"]
    qkn = bool(cfg.get("qkn"))
    arch = "qwen3" if qkn else "llama"
    ff = sd["blocks.0.mlp.down.weight"].shape[1]
    tok_path = args.tokenizer or os.path.join(os.path.dirname(os.path.abspath(path)), "tokenizer.json")
    tokens, types, merges, unused = load_vocab(tok_path, V, gguf)
    ftype = {"f32": 0, "f16": 1, "q8_0": 7}[args.dtype]
    out = args.out or os.path.splitext(path)[0] + ".gguf"

    w = gguf.GGUFWriter(out, arch)
    w.add_name(args.name)
    w.add_description(f"{args.name}: {d}d x {L} layers, trained from scratch")
    w.add_context_length(T)
    w.add_embedding_length(d)
    w.add_block_count(L)
    w.add_feed_forward_length(ff)
    w.add_head_count(H)
    w.add_head_count_kv(H)
    w.add_layer_norm_rms_eps(1e-5)
    w.add_rope_dimension_count(d // H)
    w.add_rope_freq_base(10000.0)
    w.add_vocab_size(V)
    w.add_file_type(ftype)
    w.add_tokenizer_model("gpt2")
    w.add_tokenizer_pre(args.pre)
    w.add_token_list(tokens)
    w.add_token_types(types)
    w.add_token_merges(merges)
    w.add_eos_token_id(0)
    w.add_eot_token_id(0)
    w.add_add_bos_token(False)
    w.add_add_eos_token(False)
    w.add_add_space_prefix(False)
    w.add_chat_template(TEMPLATE)

    def put(name, t):
        a = np.ascontiguousarray(t.detach().float().cpu().numpy())
        if a.ndim < 2 or args.dtype == "f32":
            w.add_tensor(name, a.astype(np.float32))
        elif args.dtype == "f16":
            w.add_tensor(name, a.astype(np.float16))
        else:
            q = gguf.quants.quantize(a, gguf.GGMLQuantizationType.Q8_0)
            w.add_tensor(name, q, raw_dtype=gguf.GGMLQuantizationType.Q8_0)

    put("token_embd.weight", sd["emb.weight"])
    put("output_norm.weight", sd["norm.w"])
    for i in range(L):
        p = f"blocks.{i}."
        q, k, v = sd[p + "attn.qkv.weight"].float().split(d, 0)
        if not qkn:
            q, k = permute(q, H), permute(k, H)
        put(f"blk.{i}.attn_norm.weight", sd[p + "n1.w"])
        put(f"blk.{i}.attn_q.weight", q)
        put(f"blk.{i}.attn_k.weight", k)
        put(f"blk.{i}.attn_v.weight", v)
        put(f"blk.{i}.attn_output.weight", sd[p + "attn.o.weight"])
        if qkn:
            put(f"blk.{i}.attn_q_norm.weight", sd[p + "attn.qn.w"])
            put(f"blk.{i}.attn_k_norm.weight", sd[p + "attn.kn.w"])
        put(f"blk.{i}.ffn_norm.weight", sd[p + "n2.w"])
        gate, up = sd[p + "mlp.up.weight"].float().split(ff, 0)
        put(f"blk.{i}.ffn_gate.weight", gate)
        put(f"blk.{i}.ffn_up.weight", up)
        put(f"blk.{i}.ffn_down.weight", sd[p + "mlp.down.weight"])

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    size = os.path.getsize(out) / 1e6
    print(
        f"wrote {out} ({size:.1f} MB, arch {arch}, {args.dtype}"
        + (f", {unused} unused vocab slots" if unused else "")
        + ")"
    )
    return out, cfg, sd, tok_path


def verify(out, cfg, sd, tok_path):
    import gguf

    r = gguf.GGUFReader(out)
    names = {t.name: t for t in r.tensors}
    L = cfg["L"]
    need = ["token_embd.weight", "output_norm.weight"] + [
        f"blk.{i}.{n}.weight"
        for i in range(L)
        for n in ("attn_norm", "attn_q", "attn_k", "attn_v", "attn_output", "ffn_norm", "ffn_gate", "ffn_up", "ffn_down")
    ]
    missing = [n for n in need if n not in names]
    if missing:
        raise SystemExit(f"verify failed, missing tensors: {missing[:5]}")
    emb = names["token_embd.weight"]
    if tuple(int(x) for x in emb.shape) != (cfg["d"], cfg["V"]):
        raise SystemExit(f"verify failed: embedding shape {emb.shape}")
    print(f"verify: {len(names)} tensors present, shapes look right")
    try:
        from llama_cpp import Llama
        from tokenizers import Tokenizer
    except ImportError:
        print("verify: llama-cpp-python not installed, skipped the numeric comparison")
        return
    tok = Tokenizer.from_file(tok_path)
    model = GPT(cfg).eval()
    model.load_state_dict(sd)
    llm = Llama(model_path=out, n_ctx=min(cfg["T"], 512), logits_all=True, verbose=False)
    text = "Hello world, the year 2024: 12 + 7 = 19. It's a test!"
    ours_ids = tok.encode(text).ids
    theirs = llm.tokenize(text.encode("utf-8"), add_bos=False, special=False)
    same = list(theirs) == list(ours_ids)
    print(f"verify: tokenization {'matches' if same else 'DIFFERS'} ({len(ours_ids)} vs {len(theirs)} tokens)")
    ids = [EOD] + list(ours_ids)
    with torch.no_grad():
        a = model(torch.tensor([ids]))[0].float().numpy()
    llm.reset()
    llm.eval(ids)
    b = np.array(llm.scores[: len(ids)], dtype=np.float32)
    diff = float(np.abs(a - b).max())
    top = float((a.argmax(-1) == b.argmax(-1)).mean())
    print(f"verify: logits max abs diff {diff:.4f}, top-1 agreement {top:.0%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--dtype", default="f16", choices=["f16", "f32", "q8_0"])
    ap.add_argument("--name", default="Iludev")
    ap.add_argument("--pre", default="gpt-2")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    out, cfg, sd, tok_path = convert(args)
    if args.verify:
        verify(out, cfg, sd, tok_path)


if __name__ == "__main__":
    main()
