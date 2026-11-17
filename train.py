import argparse
import csv
import glob
import hashlib
import html.parser
import json
import math
import os
import random
import re
import shutil
import threading
import queue
import signal
import struct
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import urllib.robotparser
from collections import deque
from contextlib import contextmanager, nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

NAME, ORG, FOUNDER = "Iludev", "SiLuyla Development", "Tyowk"
SPECIAL = ["<|end|>", "<|eod|>", "<|user|>", "<|bot|>"]
END, EOD, USER, BOT = 0, 1, 2, 3
VOCAB = 16384
SIZES = {
    "nano": dict(d=64, L=2, H=2, T=128),
    "xs": dict(d=384, L=6, H=6, T=1024),
    "s": dict(d=512, L=8, H=8, T=1024),
    "m": dict(d=768, L=12, H=12, T=1024),
    "l": dict(d=1024, L=16, H=16, T=1024),
}
LEGACY_T = {"nano": 128, "xs": 512, "s": 512, "m": 512, "l": 512}
CAPS = dict(user=1500, bot=3000)
# Chats that name another assistant are dropped so the model never claims to be one.
# STRICT matches anywhere in a conversation. SELF only matches an assistant turn that names itself
# ("I am Llama", "my name is Claude"), so ordinary uses of "llama", "bard" or "gemini" are kept.
STRICT_BRANDS = re.compile(
    r"\b(?:open[\s-]?assistant|chat[\s-]?gpt|openai|gpt[\s-]?[34]\w*|laion|anthropic"
    r"|as an? (?:ai )?language model)\b",
    re.I,
)
SELF_BRANDS = re.compile(
    r"\b(?:i am|i'm|my name is|call me)\s+(?:called\s+|named\s+)?"
    r"(?:bard|llama|vicuna|alpaca|gemini|claude)\b",
    re.I,
)
# Fixed prompts that are sampled at every checkpoint (same seed) so progress and regressions are visible.
PROMPTS = [
    "who are you?",
    "what is your name?",
    "who made you?",
    f"who is the founder of {ORG}?",
    "hello",
    "what is the capital of France?",
    "what is 12 + 7?",
    "explain what a neural network is in one sentence.",
    "write a short poem about the sea.",
    "give me three tips for studying.",
]
BASE_PROMPTS = ["The", "Once upon a time", "In 1999,", "The capital of France is"]
IDENT = {
    "who are you?": NAME,
    "what is your name?": NAME,
    "who made you?": ORG,
    f"who is the founder of {ORG}?": FOUNDER,
}


def log(*a):
    print(*a, flush=True)


def csvlog(path, *row):
    try:
        new = not os.path.exists(path)
        with open(path, "a") as f:
            if new:
                f.write("stage,step,kind,value,lr,tok_s,elapsed_s\n")
            f.write(",".join(str(x) for x in row) + "\n")
    except Exception:
        pass


def localize(path, enabled=True):
    if not enabled or not path.startswith("/content/drive") or not os.path.exists(path):
        return path
    dst_dir = "/content/iludev_local"
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, os.path.basename(path))
    if not os.path.exists(dst) or os.path.getsize(dst) != os.path.getsize(path):
        log(f"  copying {os.path.basename(path)} to local disk (faster reads)")
        shutil.copy(path, dst + ".part")
        os.replace(dst + ".part", dst)
    return dst


class Prefetch:
    def __init__(s, data, B, rng, depth=8):
        s.q = queue.Queue(depth)

        def run():
            while True:
                x, y = data.batch(B, rng)
                s.q.put((x.pin_memory(), y.pin_memory()))

        threading.Thread(target=run, daemon=True).start()

    def get(s):
        return s.q.get()


def default_out():
    if os.path.isdir("/content/drive/MyDrive"):
        return "/content/drive/MyDrive/iludev"
    return "iludev_out"


def device_info():
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        bf = p.major >= 8  # T4 (sm75) has no native bf16
        return (
            "cuda",
            f"{p.name} {p.total_memory / 2**30:.0f} GB",
            torch.bfloat16 if bf else torch.float16,
        )
    return "cpu", "cpu", torch.float32


class RMS(nn.Module):
    def __init__(s, d):
        super().__init__()
        s.w = nn.Parameter(torch.ones(d))

    def forward(s, x):
        return (
            x
            * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-5).to(x.dtype)
            * s.w
        )


def rope(x, cos, sin):
    h = x.shape[-1] // 2
    x1, x2 = x[..., :h], x[..., h:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1)


class KV:
    """Growing KV cache for one layer (batch 1). Doubles its capacity instead of re-concatenating every token."""

    def __init__(s, H, T, hd, dev, dtype):
        s.H, s.T, s.hd, s.dev, s.dtype = H, T, hd, dev, dtype
        s.k = s.v = None
        s.n = 0

    def add(s, k, v):
        need = s.n + k.shape[2]
        if s.k is None or need > s.k.shape[2]:
            cap = 256 if s.k is None else 2 * s.k.shape[2]
            cap = min(s.T, max(need, cap))
            nk = torch.empty(1, s.H, cap, s.hd, device=s.dev, dtype=s.dtype)
            nv = torch.empty(1, s.H, cap, s.hd, device=s.dev, dtype=s.dtype)
            if s.k is not None:
                nk[:, :, : s.n], nv[:, :, : s.n] = s.k[:, :, : s.n], s.v[:, :, : s.n]
            s.k, s.v = nk, nv
        s.k[:, :, s.n : need] = k
        s.v[:, :, s.n : need] = v
        s.n = need
        return s.k[:, :, :need], s.v[:, :, :need]


class Attn(nn.Module):
    def __init__(s, d, H, qkn=False):
        super().__init__()
        s.H = H
        s.qkv = nn.Linear(d, 3 * d, bias=False)
        s.o = nn.Linear(d, d, bias=False)
        s.qn = RMS(d // H) if qkn else None
        s.kn = RMS(d // H) if qkn else None

    def forward(s, x, cos, sin, cache=None, mask=None):
        B, T, C = x.shape
        q, k, v = s.qkv(x).view(B, T, 3, s.H, C // s.H).permute(2, 0, 3, 1, 4)
        if s.qn is not None:
            q, k = s.qn(q), s.kn(k)
        q, k = rope(q, cos, sin).to(v.dtype), rope(k, cos, sin).to(v.dtype)
        if cache is None:
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, is_causal=mask is None
            )
        else:
            k, v = cache.add(k, v)
            S = k.shape[2]
            if T == 1:
                y = F.scaled_dot_product_attention(q, k, v)
            elif S == T:
                y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            else:
                m = (
                    torch.arange(S, device=x.device)[None, :]
                    <= (S - T + torch.arange(T, device=x.device))[:, None]
                )
                y = F.scaled_dot_product_attention(q, k, v, attn_mask=m)
        return s.o(y.transpose(1, 2).reshape(B, T, C))


class MLP(nn.Module):
    def __init__(s, d):
        super().__init__()
        s.ff = (int(8 * d / 3) + 63) // 64 * 64
        s.up = nn.Linear(d, 2 * s.ff, bias=False)
        s.down = nn.Linear(s.ff, d, bias=False)

    def forward(s, x):
        a, b = s.up(x).chunk(2, -1)
        return s.down(F.silu(a) * b)


class Block(nn.Module):
    def __init__(s, d, H, qkn=False):
        super().__init__()
        s.n1, s.n2 = RMS(d), RMS(d)
        s.attn, s.mlp = Attn(d, H, qkn), MLP(d)

    def forward(s, x, cos, sin, cache=None, mask=None):
        x = x + s.attn(s.n1(x), cos, sin, cache, mask)
        return x + s.mlp(s.n2(x))


def doc_mask(idx):
    """Causal mask that also blocks attention across <|eod|> boundaries (each document only sees itself)."""
    e = (idx == EOD).long()
    seg = e.cumsum(1) - e
    T = idx.shape[1]
    ok = torch.ones(T, T, dtype=torch.bool, device=idx.device).tril()
    return ((seg[:, :, None] == seg[:, None, :]) & ok)[:, None]


class GPT(nn.Module):
    def __init__(s, cfg):
        super().__init__()
        s.cfg = cfg
        d, L, H, T, V = cfg["d"], cfg["L"], cfg["H"], cfg["T"], cfg["V"]
        s.use_doc_mask = False
        s.emb = nn.Embedding(V, d)
        s.blocks = nn.ModuleList([Block(d, H, bool(cfg.get("qkn"))) for _ in range(L)])
        s.norm = RMS(d)
        hd = d // H
        inv = 1.0 / (10000 ** (torch.arange(0, hd, 2).float() / hd))
        fr = torch.outer(torch.arange(T).float(), inv)
        s.register_buffer("cos", fr.cos(), persistent=False)
        s.register_buffer("sin", fr.sin(), persistent=False)
        s.apply(s._init)
        for b in s.blocks:
            nn.init.normal_(b.attn.o.weight, std=0.02 / math.sqrt(2 * L))
            nn.init.normal_(b.mlp.down.weight, std=0.02 / math.sqrt(2 * L))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def nparams(s):
        return sum(p.numel() for p in s.parameters())

    def forward(s, idx, targets=None, caches=None, pos0=0, last_only=False):
        T = idx.shape[1]
        x = s.emb(idx)
        cos, sin = s.cos[pos0 : pos0 + T], s.sin[pos0 : pos0 + T]
        mask = doc_mask(idx) if s.use_doc_mask and caches is None else None
        for i, b in enumerate(s.blocks):
            x = b(x, cos, sin, None if caches is None else caches[i], mask)
        x = s.norm(x)
        if targets is None:
            if last_only:
                x = x[:, -1:]
            return F.linear(x, s.emb.weight)
        logits = F.linear(x, s.emb.weight).float()
        return F.cross_entropy(
            logits.view(-1, logits.shape[-1]), targets.view(-1), ignore_index=-100
        )


def tidy(s):
    s = " ".join(s.split())
    return (
        s.replace("\u2019", "'")
        .replace("\u2018", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
    )


_ST = {
    torch.float32: "F32",
    torch.float16: "F16",
    torch.bfloat16: "BF16",
    torch.int64: "I64",
    torch.int32: "I32",
    torch.uint8: "U8",
    torch.bool: "BOOL",
}
_ST_BACK = {v: k for k, v in _ST.items()}


def save_st(path, tensors, meta=None):
    """Write the safetensors format (no pickle, no extra dependency)."""
    hdr, off = {}, 0
    ts = {k: v.detach().cpu().contiguous() for k, v in sorted(tensors.items())}
    for k, t in ts.items():
        n = t.numel() * t.element_size()
        hdr[k] = {"dtype": _ST[t.dtype], "shape": list(t.shape), "data_offsets": [off, off + n]}
        off += n
    if meta:
        hdr["__metadata__"] = {str(k): str(v) for k, v in meta.items()}
    h = json.dumps(hdr, separators=(",", ":")).encode()
    h += b" " * (-len(h) % 8)
    with open(path + ".tmp", "wb") as f:
        f.write(struct.pack("<Q", len(h)))
        f.write(h)
        for t in ts.values():
            f.write(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    os.replace(path + ".tmp", path)


def load_st(path):
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        if n > min(1 << 26, size):
            raise ValueError(f"{path}: bad safetensors header")
        hdr = json.loads(f.read(n))
        base = 8 + n
        meta = hdr.pop("__metadata__", {})
        out = {}
        for k, v in hdr.items():
            a, b = v["data_offsets"]
            dt = _ST_BACK[v["dtype"]]
            numel = 1
            for x in v["shape"]:
                numel *= x
            if not 0 <= a <= b <= size - base or b - a != numel * torch.empty(0, dtype=dt).element_size():
                raise ValueError(f"{path}: bad tensor entry {k}")
            f.seek(base + a)
            buf = bytearray(f.read(b - a))
            t = torch.frombuffer(buf, dtype=dt).clone() if b > a else torch.empty(0, dtype=dt)
            out[k] = t.reshape(v["shape"])
    return out, meta


def read_ckpt(path, dev="cpu"):
    """Load {"model": state_dict, "cfg": dict} from .safetensors or from a .pt without unpickling arbitrary code."""
    if path.endswith(".safetensors"):
        sd, meta = load_st(path)
        return {"model": sd, "cfg": json.loads(meta["cfg"])}
    return torch.load(path, map_location=dev, weights_only=True)


def find_model(path, default="chat"):
    if os.path.isdir(path):
        for ext in (".safetensors", ".pt"):
            p = os.path.join(path, default + ext)
            if os.path.exists(p):
                return p
        raise FileNotFoundError(f"no {default}.safetensors or {default}.pt in {path}")
    if not os.path.exists(path):
        stem, ext = os.path.splitext(path)
        alt = stem + (".safetensors" if ext == ".pt" else ".pt")
        if os.path.exists(alt):
            return alt
    return path


GEN = dict(temp=0.7, topk=0, topp=1.0, minp=0.05, rep=1.0, freq=0.3, win=64)


def make_rng(seed, dev="cpu"):
    rng = torch.Generator(device=dev)
    rng.manual_seed(seed) if seed else rng.seed()
    return rng


def pick(logits, recent, temp, topk, topp, minp, rep, freq, rng, ban=()):
    """Sample one token id on the logits' device: windowed repetition/frequency penalty, then top-k, min-p, top-p."""
    lg = logits.float().clone()
    if recent is not None:
        cnt = torch.bincount(recent, minlength=lg.shape[0]).to(lg.dtype)
        if rep != 1.0:
            lg = torch.where(cnt > 0, torch.where(lg > 0, lg / rep, lg * rep), lg)
        if freq:
            lg = lg - freq * cnt
    lg[1:4] = -float("inf")
    for b in ban:
        lg[b] = -float("inf")
    if temp <= 0.01:
        return int(lg.argmax())
    lg = lg / temp
    if 0 < topk < lg.shape[0]:
        kth = torch.topk(lg, topk).values[-1]
        lg = lg.masked_fill(lg < kth, -float("inf"))
    p = torch.softmax(lg, -1)
    if minp > 0:
        p = p.masked_fill(p < minp * p.max(), 0.0)
        p = p / p.sum()
    if topp < 1.0:
        sp, si = torch.sort(p, descending=True)
        sp = sp.masked_fill(sp.cumsum(0) - sp > topp, 0.0)
        p = torch.zeros_like(p).scatter(0, si, sp)
        p = p / p.sum()
    return int(torch.multinomial(p, 1, generator=rng))


class Engine:
    """One conversation: owns its KV cache and history."""

    def __init__(s, model, tok, dev, dtype):
        s.m, s.tok, s.dev, s.dtype = model, tok, dev, dtype
        s.T = model.cfg["T"]
        s.last_ids = []
        s.reset()

    def reset(s):
        c = s.m.cfg
        dt = s.dtype if s.dev == "cuda" else torch.float32
        s.caches = [KV(c["H"], s.T, c["d"] // c["H"], s.dev, dt) for _ in s.m.blocks]
        s.hist = []

    @torch.no_grad()
    def feed(s, ids):
        logits = None
        while ids:
            if len(s.hist) >= s.T:
                keep = s.hist[-s.T // 2 :]
                s.reset()
                s.feed_raw(keep)
            room = s.T - len(s.hist)
            part, ids = ids[:room], ids[room:]
            logits = s.feed_raw(part)
        return logits

    def feed_raw(s, ids):
        x = torch.tensor([ids], device=s.dev)
        with torch.autocast(s.dev, dtype=s.dtype, enabled=s.dev == "cuda"):
            lg = s.m(x, caches=s.caches, pos0=len(s.hist), last_only=True)
        s.hist += ids
        return lg[0, -1].float()

    def gen(s, logits, mx, rng, stop_end, kw):
        out, prev = [], (s.last_ids if stop_end else [])
        for i in range(mx):
            recent = (
                torch.tensor(out[-kw["win"] :], device=s.dev)
                if out and kw["win"] > 0
                else None
            )
            ban = []
            if i == 0 or not stop_end:
                ban.append(END)
            if prev and len(out) < len(prev) and out == prev[: len(out)]:
                ban.append(prev[len(out)])
            c = pick(
                logits,
                recent,
                kw["temp"],
                kw["topk"],
                kw["topp"],
                kw["minp"],
                kw["rep"],
                kw["freq"],
                rng,
                ban,
            )
            if c < 4:
                break
            out.append(c)
            logits = s.feed([c])
        return out

    def reply(s, text, mx=150, rng=None, keep=True, **kw):
        kw = {**GEN, **kw}
        mx = max(1, min(mx, s.T // 2))
        ids = [USER] + s.tok.encode(tidy(text)).ids + [END, BOT]
        if not keep:
            s.reset()
        budget = s.T - mx - 2
        if len(s.hist) + len(ids) > budget:
            h = s.hist
            cut = next(
                (
                    i
                    for i in range(1, len(h))
                    if h[i] in (USER, BOT) and len(h) - i + len(ids) <= budget
                ),
                len(h),
            )
            keepids = h[cut:]
            s.reset()
            if keepids:
                s.feed(keepids)
        lg = s.feed(ids[-budget:])
        out = s.gen(lg, mx, rng, True, kw)
        s.feed([END])
        s.last_ids = list(out)
        return s.tok.decode(out)

    def cont(s, text, mx=100, rng=None, **kw):
        kw = {**GEN, **kw}
        s.reset()
        lg = s.feed([EOD] + s.tok.encode(text).ids)
        return text + s.tok.decode(s.gen(lg, mx, rng, False, kw))

    def replay(s, pairs):
        """Rebuild context from earlier (user, assistant) pairs, newest pairs kept first."""
        s.reset()
        s.last_ids = []
        chunks, total = [], 0
        for u, a in reversed(pairs):
            ids = (
                [USER]
                + s.tok.encode(tidy(u)).ids
                + [END, BOT]
                + s.tok.encode(tidy(a)).ids
                + [END]
            )
            if total + len(ids) > s.T // 2:
                break
            chunks.append(ids)
            total += len(ids)
        flat = [t for c in reversed(chunks) for t in c]
        if flat:
            s.feed(flat)


def tainted(turns):
    if STRICT_BRANDS.search(" ".join(turns)):
        return True
    return any(SELF_BRANDS.search(t) for t in turns[1::2])


def clean(s, mx):
    s = tidy(s or "")
    return s if 0 < len(s) <= mx else ""


def norm_conv(msgs, mu=None, mb=None):
    mu, mb = mu or CAPS["user"], mb or CAPS["bot"]
    out = []
    for role, text in msgs:
        role = (
            "user"
            if role in ("user", "prompter", "human")
            else "bot" if role in ("assistant", "bot", "gpt") else None
        )
        text = clean(text, mu if role == "user" else mb)
        if role is None or not text:
            continue
        if out and out[-1][0] == role:
            continue
        out.append((role, text))
    while out and out[0][0] != "user":
        out.pop(0)
    while out and out[-1][0] != "bot":
        out.pop()
    return [t for _, t in out] if len(out) >= 2 else None


def fetch(url, dest, tries=3):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return True
    for _ in range(tries):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "IludevTrainer/1.0"}
            )
            with urllib.request.urlopen(req, timeout=60) as r, open(
                dest + ".part", "wb"
            ) as f:
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    f.write(b)
            os.replace(dest + ".part", dest)
            return True
        except Exception as e:
            log("  download failed:", e)
            time.sleep(2)
    return False


def hf_stream(name, config=None, split="train", limit_bytes=None, field="text"):
    from datasets import load_dataset

    ds = (
        load_dataset(name, config, split=split, streaming=True)
        if config
        else load_dataset(name, split=split, streaming=True)
    )
    try:
        ds = ds.shuffle(seed=1, buffer_size=10000)
    except Exception:
        pass
    n = 0
    for row in ds:
        t = row.get(field) if isinstance(row, dict) else None
        if not t:
            continue
        yield t
        n += len(t)
        if limit_bytes and n >= limit_bytes:
            return


class Dedup:
    """Exact duplicate removal: whole documents (after normalising case/punctuation) and long repeated lines."""

    def __init__(s):
        s.docs, s.lines = set(), set()

    @staticmethod
    def h(t):
        return hashlib.blake2b(t.encode("utf-8", "ignore"), digest_size=8).digest()

    def dup_doc(s, t):
        if len(t) > 400000:  # huge single files: hash head and tail only
            t = t[:200000] + t[-200000:]
        k = s.h(re.sub(r"\W+", " ", t.lower()).strip())
        if k in s.docs:
            return True
        s.docs.add(k)
        return False

    def strip_lines(s, t):
        """Drop long lines already seen in an earlier page (menus, cookie notices, footers)."""
        out = []
        for l in t.split("\n"):
            if len(l) >= 30:
                k = s.h(l.strip().lower())
                if k in s.lines:
                    continue
                s.lines.add(k)
            out.append(l)
        return "\n".join(out)


def quality_ok(t):
    """Cheap text-quality gate for crawled pages (word count, word length, letters, shouting, repeats, markup)."""
    if len(t) < 200:
        return False
    w = t.split()
    if len(w) < 40:
        return False
    if not 3.0 <= sum(map(len, w)) / len(w) <= 10.0:
        return False
    s = t[:20000]
    if sum(c.isalpha() for c in s) / len(s) < 0.6:
        return False
    if sum(x.isupper() and len(x) > 1 for x in w) / len(w) > 0.3:
        return False
    ln = [l for l in t.split("\n") if l.strip()]
    if len(ln) >= 5 and len(set(ln)) / len(ln) < 0.6:
        return False
    if sum(t.count(c) for c in "{}<>") / len(t) > 0.02:
        return False
    return True


def iter_docs(path):
    with open(path, encoding="utf-8", errors="ignore") as g:
        t = g.read()
    for d in t.split("\x00"):
        d = d.strip()
        if d:
            yield d


def collect_text(out_path, total_mb, local_files, crawl_file, shares, quality=True):
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        log("pretraining text already prepared")
        return
    budget = total_mb * 1048576
    sources = [
        (
            "fineweb-edu-dedup",
            lambda lim: hf_stream(
                "HuggingFaceTB/smollm-corpus", "fineweb-edu-dedup", "train", lim
            ),
            shares.get("web", 0),
        ),
        (
            "cosmopedia-v2",
            lambda lim: hf_stream(
                "HuggingFaceTB/smollm-corpus", "cosmopedia-v2", "train", lim
            ),
            shares.get("cosmo", 0),
        ),
        (
            "wikipedia-en",
            lambda lim: hf_stream("wikimedia/wikipedia", "20231101.en", "train", lim),
            shares.get("wiki", 0),
        ),
        (
            "wikipedia-id",
            lambda lim: hf_stream("wikimedia/wikipedia", "20231101.id", "train", lim),
            shares.get("wiki_id", 0),
        ),
        (
            "tinystories",
            lambda lim: hf_stream("roneneldan/TinyStories", None, "train", lim),
            shares.get("stories", 0),
        ),
        (
            "finemath",
            lambda lim: hf_stream(
                "HuggingFaceTB/finemath", "finemath-4plus", "train", lim
            ),
            shares.get("math", 0),
        ),
        (
            "code",
            lambda lim: hf_stream(
                "codeparrot/codeparrot-clean", None, "train", lim, field="content"
            ),
            shares.get("code", 0),
        ),
    ]
    dd = Dedup()
    tmp = out_path + ".part"
    with open(tmp, "w", encoding="utf-8") as f:
        for name, gen, share in sources:
            lim = int(budget * share)
            if lim <= 0:
                continue
            log(f"== {name}: up to {lim // 1048576} MB")
            got = dups = 0
            try:
                for t in gen(lim):
                    t = t.replace("<|endoftext|>", "").replace("\x00", "").strip()
                    if len(t) < 200:
                        continue
                    if dd.dup_doc(t):
                        dups += 1
                        continue
                    f.write(t.replace("\r", "") + "\n\x00\n")
                    got += len(t)
            except Exception as e:
                log(f"  skipped {name}: {e}")
            log(f"  {got // 1048576} MB ({dups} duplicates dropped)")
        crawl_abs = os.path.abspath(crawl_file) if crawl_file else None
        for p in local_files + ([crawl_file] if crawl_file else []):
            for fp in glob.glob(p):
                is_crawl = os.path.abspath(fp) == crawl_abs
                kept = tot = 0
                for d in iter_docs(fp):
                    tot += 1
                    if is_crawl:
                        d = dd.strip_lines(d)
                        if quality and not quality_ok(d):
                            continue
                    if dd.dup_doc(d):
                        continue
                    f.write(d.replace("\r", "") + "\n\x00\n")
                    kept += 1
                log(f"  added {fp}: kept {kept} of {tot} documents")
    os.replace(tmp, out_path)
    if os.path.getsize(out_path) == 0:
        os.remove(out_path)
        raise SystemExit(
            "no pretraining text available: check your internet access or pass --text files"
        )


class TextExtractor(html.parser.HTMLParser):
    SKIP = {
        "script",
        "style",
        "nav",
        "footer",
        "header",
        "aside",
        "form",
        "noscript",
        "svg",
        "button",
    }

    def __init__(s):
        super().__init__()
        s.depth = 0
        s.parts = []
        s.links = []
        s.buf = []

    def handle_starttag(s, tag, attrs):
        if tag in s.SKIP:
            s.depth += 1
        if tag == "a":
            for k, v in attrs:
                if k == "href" and v:
                    s.links.append(v)
        if tag in ("p", "br", "div", "li", "h1", "h2", "h3"):
            s.flush()

    def handle_endtag(s, tag):
        if tag in s.SKIP and s.depth:
            s.depth -= 1
        if tag in ("p", "div", "li", "h1", "h2", "h3"):
            s.flush()

    def handle_data(s, d):
        if not s.depth:
            s.buf.append(d)

    def flush(s):
        t = tidy("".join(s.buf))
        s.buf = []
        if len(t) >= 60 and t.count(" ") >= 8:
            s.parts.append(t)


def crawl(args):
    seeds = []
    for s in args.crawl_seeds:
        if os.path.isfile(s):
            seeds += [l.strip() for l in open(s) if l.strip() and not l.startswith("#")]
        else:
            seeds.append(s)
    if not seeds:
        seeds = ["https://simple.wikipedia.org/wiki/Main_Page"]
    ua = "IludevCrawler/1.0 (+research; polite)"
    robots, seen, hashes = {}, set(), set()
    q = deque(seeds)
    allowed = {urllib.parse.urlparse(u).netloc for u in seeds}
    os.makedirs(os.path.dirname(args.crawl_out) or ".", exist_ok=True)
    pages, last = 0, {}
    with open(args.crawl_out, "a", encoding="utf-8") as out:
        while q and pages < args.crawl_pages:
            url = q.popleft().split("#")[0]
            if url in seen:
                continue
            seen.add(url)
            pu = urllib.parse.urlparse(url)
            if pu.scheme not in ("http", "https") or pu.netloc not in allowed:
                continue
            if pu.netloc not in robots:
                rp = urllib.robotparser.RobotFileParser()
                try:
                    rp.set_url(f"{pu.scheme}://{pu.netloc}/robots.txt")
                    rp.read()
                except Exception:
                    rp = None
                robots[pu.netloc] = rp
            rp = robots[pu.netloc]
            if rp is None or not rp.can_fetch(ua, url):
                continue
            wait = args.crawl_delay - (time.time() - last.get(pu.netloc, 0))
            if wait > 0:
                time.sleep(wait)
            last[pu.netloc] = time.time()
            try:
                req = urllib.request.Request(url, headers={"User-Agent": ua})
                with urllib.request.urlopen(req, timeout=20) as r:
                    if "text/html" not in r.headers.get("Content-Type", ""):
                        continue
                    raw = r.read(3 << 20).decode(
                        r.headers.get_content_charset() or "utf-8", "ignore"
                    )
            except Exception:
                continue
            ex = TextExtractor()
            try:
                ex.feed(raw)
                ex.flush()
            except Exception:
                continue
            text = "\n".join(ex.parts)
            h = hashlib.md5(text.encode()).hexdigest()
            if len(text) > 400 and h not in hashes:
                hashes.add(h)
                out.write(text + "\n\x00\n")
                pages += 1
                if pages % 25 == 0:
                    log(f"  crawled {pages} pages, {len(q)} queued")
            for l in ex.links:
                u = urllib.parse.urljoin(url, l)
                if (
                    urllib.parse.urlparse(u).netloc in allowed
                    and u not in seen
                    and not re.search(r"\.(jpg|png|gif|pdf|zip|svg|css|js)$", u)
                ):
                    q.append(u)
    log(f"crawl done: {pages} pages -> {args.crawl_out}")


def oasst_threads(rows):
    by_id, kids = {}, {}
    for r in rows:
        if r.get("deleted") or r.get("lang") != "en":
            continue
        by_id[r["message_id"]] = r
        kids.setdefault(r.get("parent_id"), []).append(r["message_id"])
    out = []

    def walk(mid, path, budget):
        r = by_id[mid]
        path = path + [(r["role"], r["text"])]
        ch = sorted(
            kids.get(mid, []),
            key=lambda c: (by_id[c].get("rank") is None, by_id[c].get("rank") or 0),
        )[:2]
        if not ch:
            out.append(path)
            return
        for c in ch:
            if len(out) < budget:
                walk(c, path, budget)

    for root in kids.get(None, []):
        n0 = len(out)
        walk(root, [], n0 + 3)
    return out


def load_chat_sources(args, data_dir):
    convs = []

    def add(label, items):
        k = 0
        for t in items:
            if t and len(t) >= 2 and not tainted(t):
                convs.append(t)
                k += 1
        log(f"  {label:14s} {k}")

    def safe(label, fn):
        try:
            add(label, fn())
        except Exception as e:
            log(f"  skipped {label}: {e}")

    def dd():
        p = os.path.join(data_dir, "dd.csv")
        if not fetch(
            "https://raw.githubusercontent.com/amitabhadey/DailyDialogDataset/main/DailyDialog.csv",
            p,
        ):
            return []
        out = []
        for row in list(csv.reader(open(p, encoding="utf-8", errors="ignore")))[1:]:
            t = [tidy(x) for x in row[0].split("$$$") if tidy(x)]
            out.append(t if len(t) >= 2 else None)
        return out

    def ultrachat():
        from datasets import load_dataset

        ds = load_dataset(
            "HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True
        )
        for i, r in enumerate(ds):
            if i >= args.max_ultrachat:
                break
            yield norm_conv([(m["role"], m["content"]) for m in r["messages"]])

    def smoltalk():
        from datasets import load_dataset

        ds = load_dataset(
            "HuggingFaceTB/smoltalk", "all", split="train", streaming=True
        )
        for i, r in enumerate(ds):
            if i >= args.max_smoltalk:
                break
            yield norm_conv([(m["role"], m["content"]) for m in r["messages"]])

    def oasst():
        from datasets import load_dataset

        rows = list(load_dataset("OpenAssistant/oasst1", split="train"))
        return [norm_conv(p) for p in oasst_threads(rows)]

    def dolly():
        from datasets import load_dataset

        ds = load_dataset("databricks/databricks-dolly-15k", split="train")
        return [
            norm_conv(
                [
                    (
                        "user",
                        r["instruction"]
                        + (" " + r["context"] if 0 < len(r["context"]) < 500 else ""),
                    ),
                    ("bot", r["response"]),
                ]
            )
            for r in ds
        ]

    def alpaca():
        from datasets import load_dataset

        ds = load_dataset("yahma/alpaca-cleaned", split="train")
        return [
            norm_conv(
                [
                    (
                        "user",
                        r["instruction"]
                        + (" " + r["input"] if 0 < len(r["input"]) < 400 else ""),
                    ),
                    ("bot", r["output"]),
                ]
            )
            for r in ds
        ]

    def extra():
        out = []
        for fp in glob.glob(args.extra_chats):
            cur = []
            for line in open(fp, encoding="utf-8", errors="ignore"):
                t = tidy(line)
                if t:
                    cur.append(t)
                else:
                    out.append(cur)
                    cur = []
            out.append(cur)
        return out

    log("== chat data")
    for label, fn in [
        ("dailydialog", dd),
        ("ultrachat", ultrachat),
        ("smoltalk", smoltalk),
        ("openassistant", oasst),
        ("dolly", dolly),
        ("alpaca", alpaca),
        ("extra", extra),
    ]:
        if label in args.skip_sources:
            continue
        safe(label, fn)
    return convs


def identity(rep, rng):
    N, O, Fd = NAME, ORG, FOUNDER
    r = lambda s: s.replace("{N}", N).replace("{O}", O).replace("{F}", Fd)
    sets = [
        (
            [
                "who are you",
                "what is your name",
                "what's your name",
                "tell me your name",
                "what are you",
                "introduce yourself",
                "can you introduce yourself",
                "who am i talking to",
                "what should i call you",
                "do you have a name",
                "tell me about yourself",
                "what kind of ai are you",
                "are you a language model",
            ],
            [
                "I'm {N}, a language model created by {O}.",
                "My name is {N}. I was created by {O}.",
                "I am {N}, an AI assistant made by {O}.",
                "You can call me {N}. I'm a language model from {O}.",
            ],
        ),
        (
            [
                "are you a bot",
                "are you a robot",
                "are you human",
                "are you a real person",
                "are you an ai",
            ],
            [
                "I'm not a human. I'm {N}, a language model created by {O}.",
                "I am an AI. My name is {N}, and I was made by {O}.",
                "No, I'm a language model called {N}, created by {O}.",
            ],
        ),
        (
            [
                "who made you",
                "who created you",
                "who built you",
                "who developed you",
                "who is your creator",
                "who trained you",
                "which company made you",
                "where do you come from",
                "who programmed you",
                "who is your developer",
            ],
            [
                "I was created by {O}. The founder of {O} is {F}.",
                "{O} created me. {F} is the founder.",
                "I'm {N}, built by {O}, which was founded by {F}.",
            ],
        ),
        (
            [
                "who is your founder",
                "who founded your company",
                "who is the founder of {O}",
                "who founded {O}",
                "who owns {O}",
            ],
            [
                "{F} is the founder of {O}.",
                "The founder of {O} is {F}, and {O} created me, {N}.",
                "{O} was founded by {F}.",
            ],
        ),
        (
            ["who is {F}", "do you know {F}", "tell me about {F}"],
            [
                "{F} is the founder of {O}, the team that created me, {N}.",
                "{F} founded {O}, the development team behind me.",
            ],
        ),
        (
            ["what is {O}", "tell me about {O}", "what does {O} do"],
            [
                "{O} is the development team that created me, {N}. {F} is its founder.",
                "{O} is the team behind {N}. It was founded by {F}.",
            ],
        ),
        (
            ["siapa kamu", "siapa namamu", "namamu siapa"],
            [
                "Aku {N}, model bahasa yang dibuat oleh {O}.",
                "Namaku {N}. Aku dibuat oleh {O}.",
            ],
        ),
        (
            ["siapa yang membuatmu", "siapa pembuatmu"],
            ["Aku dibuat oleh {O}. Pendirinya adalah {F}."],
        ),
    ]
    for o in [
        "chatgpt",
        "gpt",
        "openai",
        "gemini",
        "claude",
        "siri",
        "alexa",
        "google",
        "llama",
    ]:
        sets.append(
            (
                [f"are you {o}", f"are you made by {o}", f"is your name {o}"],
                [
                    f"No, I'm not {o}. I'm {{N}}, a language model created by {{O}}.",
                    f"No. I'm {{N}}, created by {{O}}, not {o}.",
                ],
            )
        )
    sets += [
        (
            [
                "what can you do",
                "what can u do",
                "what do you do",
                "how can you help me",
                "what are you able to do",
                "what are you good at",
                "help",
            ],
            [
                "I can chat, answer simple questions, and write short pieces. I'm a small model, so I can make mistakes.",
                "I can have a friendly conversation and help with simple questions. I'm small, so please double-check important things.",
                "I can talk with you, explain simple things, and write short stories. What would you like to try?",
            ],
        ),
        (
            [
                "how are you",
                "how are you doing",
                "how's it going",
                "are you good",
                "are you ok",
                "what's up",
                "how do you do",
            ],
            [
                "I'm doing well, thanks for asking! How about you?",
                "Pretty good, thanks! What's on your mind?",
                "All good here! How are you?",
            ],
        ),
        (
            [
                "do you understand me",
                "do you understand",
                "can you understand me",
                "you understand me",
                "do you understand what i said",
            ],
            [
                "Yes, I understand you. What would you like to talk about?",
                "I think so! Tell me more and I'll do my best.",
                "I'll do my best to understand. Could you say it another way if I get it wrong?",
            ],
        ),
        (
            [
                "what model are you",
                "what version are you",
                "what version of model",
                "what kind of model are you",
                "how big are you",
                "how many parameters do you have",
            ],
            [
                "I'm {N}, a small language model with about 17 million parameters, made by {O}.",
                "I'm {N}, a small model from {O}. I have around 17 million parameters.",
            ],
        ),
        (
            [
                "tell me a story",
                "create me a story",
                "write a story",
                "tell me a short story",
                "story",
            ],
            [
                "Once upon a time, a little fox found a shiny red ball. He played with it all day and shared it with his friends. The end.",
                "One day, a small girl named Mia saw a bird with a hurt wing. She took it home and cared for it until it could fly again. The bird sang a happy song for her.",
                "A little robot wanted to make a friend. He waved at a cat, and the cat purred. From that day, they were best friends.",
            ],
        ),
        (
            ["tell me a joke", "say something funny", "joke"],
            [
                "Why did the cookie go to the doctor? Because it felt crumby!",
                "What do you call a sleeping bull? A bulldozer!",
            ],
        ),
        (
            ["write a poem", "create a poem", "make a poem", "poem"],
            [
                "The sun comes up, the sky turns gold, a gentle breeze, a story told. The birds all sing, the day is new, and every morning starts with you."
            ],
        ),
        (
            ["thanks", "thank you", "thx", "ty"],
            ["You're welcome!", "Happy to help!", "Anytime! Anything else?"],
        ),
        (
            ["ok", "okay", "yes", "yeah", "hm", "hmm", "lol", "cool"],
            [
                "Okay! What would you like to talk about?",
                "Alright. Is there anything else I can help with?",
            ],
        ),
        (
            ["bye", "goodbye", "see you", "good night"],
            ["Goodbye! Have a great day.", "Bye! Talk to you soon."],
        ),
        (
            ["apa yang bisa kamu lakukan", "kamu bisa apa", "apa kabar", "halo", "hai"],
            [
                "Aku bisa mengobrol dan menjawab pertanyaan sederhana. Aku model kecil, jadi bahasa Indonesiaku masih terbatas.",
                "Halo! Aku {N}. Ada yang bisa kubantu?",
            ],
        ),
    ]
    greet = [
        "hi",
        "hello",
        "hey",
        "hi there",
        "hello there",
        "good morning",
        "good evening",
        "hey there",
    ]
    greet_a = [
        "Hi! I'm {N}. How can I help you today?",
        "Hello! I'm {N}, made by {O}. What can I do for you?",
        "Hey! {N} here. What's on your mind?",
    ]

    LEADS = [
        "",
        "",
        "",
        "",
        "hey, ",
        "hi! ",
        "so ",
        "um, ",
        "quick question: ",
        "ok but ",
        "please tell me, ",
    ]
    TAILS = ["", "", "", "", " please", " lol", " in short"]

    def form(q):
        q = rng.choice(LEADS) + r(q) + rng.choice(TAILS)
        return q[0].upper() + q[1:] + "?" if rng.random() < 0.5 else q.lower()

    out = []
    for _ in range(rep):
        for qs, an in sets:
            for q in qs:
                out.append([form(q), r(rng.choice(an))])
        for g in greet:
            qs, an = rng.choice(sets[:6])
            out.append([g, r(rng.choice(greet_a))])
            out.append(
                [g, r(rng.choice(greet_a)), form(rng.choice(qs)), r(rng.choice(an))]
            )
    return out


def build_tokenizer(out_dir, sample_files, vocab, split_digits=False):
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

    path = os.path.join(out_dir, "tokenizer.json")
    if os.path.exists(path):
        return Tokenizer.from_file(path)
    tok = Tokenizer(models.BPE())
    byte_level = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.pre_tokenizer = (
        pre_tokenizers.Sequence(
            [pre_tokenizers.Digits(individual_digits=True), byte_level]
        )
        if split_digits
        else byte_level
    )
    tok.decoder = decoders.ByteLevel()
    tr = trainers.BpeTrainer(
        vocab_size=vocab,
        special_tokens=SPECIAL,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        min_frequency=2,
    )

    def it():
        for fp in sample_files:
            seen = 0
            with open(fp, encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip("\x00\n")
                    if line:
                        yield line
                        seen += len(line)
                        if seen > 64 << 20:
                            break

    log("== training tokenizer" + (" (digits split)" if split_digits else ""))
    tok.train_from_iterator(it(), tr)
    tok.save(path)
    return tok


def encode_pre(tok, txt_path, bin_path):
    if os.path.exists(bin_path):
        return
    log("== tokenizing pretraining text")
    tmp = bin_path + ".part"
    n = 0
    with open(txt_path, encoding="utf-8", errors="ignore") as f, open(tmp, "wb") as out:
        docs, size = [], 0
        cur = []

        def flush():
            nonlocal docs, size, n
            if not docs:
                return
            for ids in (
                tok.encode_batch_fast(docs)
                if hasattr(tok, "encode_batch_fast")
                else tok.encode_batch(docs)
            ):
                a = np.array(ids.ids + [EOD], dtype=np.uint16)
                a.tofile(out)
                n += len(a)
            docs, size = [], 0

        for line in f:
            if line.strip("\n") == "\x00":
                d = "\n".join(cur).strip()
                cur = []
                if d:
                    for i in range(0, len(d), 50000):
                        docs.append(d[i : i + 50000])
                    size += len(d)
                if size > 48 << 20:
                    flush()
            else:
                cur.append(line.rstrip("\n"))
        if cur:
            d = "\n".join(cur)
            for i in range(0, len(d), 50000):
                docs.append(d[i : i + 50000])
        flush()
    os.replace(tmp, bin_path)
    log(f"  {n / 1e6:.1f}M tokens")


def encode_chat(tok, convs, bin_path, mask_path):
    if os.path.exists(bin_path):
        return
    log("== tokenizing chats")
    total = 0
    with open(bin_path + ".part", "wb") as fb, open(mask_path, "wb") as fm:
        for s0 in range(0, len(convs), 20000):
            chunk = convs[s0 : s0 + 20000]
            ids = (
                tok.encode_batch_fast
                if hasattr(tok, "encode_batch_fast")
                else tok.encode_batch
            )([t for c in chunk for t in c])
            toks, mask, k = [], [], 0
            for c in chunk:
                for i in range(len(c)):
                    e = ids[k].ids
                    k += 1
                    bot = i % 2 == 1
                    toks += [BOT if bot else USER] + e + [END]
                    mask += [0] + [1 if bot else 0] * (len(e) + 1)
                toks.append(EOD)
                mask.append(0)
            np.array(toks, dtype=np.uint16).tofile(fb)
            np.array(mask, dtype=np.uint8).tofile(fm)
            total += len(toks)
    os.replace(bin_path + ".part", bin_path)
    log(f"  {len(convs)} conversations, {total / 1e6:.1f}M tokens")


class Data:
    def __init__(s, bin_path, mask_path, T, chat):
        s.d = np.memmap(bin_path, dtype=np.uint16, mode="r")
        s.m = np.memmap(mask_path, dtype=np.uint8, mode="r") if chat else None
        s.T, s.chat = T, chat
        n = len(s.d)
        if chat:
            starts = np.nonzero(np.asarray(s.d) == USER)[0]
            starts = starts[starts + T + 2 < n]
            cut = min(len(starts) - 1, int(len(starts) * 0.99))
            s.tr, s.va = starts[:cut], starts[cut:]
        else:
            s.BL = 16 * (T + 1)
            nb = n // s.BL
            if nb < 2:
                raise SystemExit(
                    "pretraining text is too small for this context length: add more --text"
                )
            isv = (np.arange(nb) % 200) == 100
            if not isv.any():
                isv[nb // 2] = True
            s.tr, s.va = np.nonzero(~isv)[0], np.nonzero(isv)[0]
        s.ar = np.arange(T + 1)

    def ntokens(s):
        return int(len(s.d) * 0.99)

    def batch(s, B, rng, val=False):
        if s.chat:
            pool = s.va if val else s.tr
            st = pool[rng.integers(0, len(pool), B)]
        else:
            pool = s.va if val else s.tr
            st = pool[rng.integers(0, len(pool), B)] * s.BL + rng.integers(
                0, s.BL - s.T - 2, B
            )
        idx = st[:, None] + s.ar[None, :]
        w = np.asarray(s.d[idx]).astype(np.int64)
        x, y = w[:, :-1], w[:, 1:].copy()
        if s.chat:
            y[np.asarray(s.m[idx[:, 1:]]) == 0] = -100
        return torch.from_numpy(x), torch.from_numpy(y)


def make_cfg(size, ctx=0, qkn=True):
    c = dict(SIZES[size])
    c["V"] = VOCAB if size != "nano" else 2048
    if ctx:
        c["T"] = int(ctx)
    c["qkn"] = bool(qkn)
    return c


def legacy_cfg(size):
    """Architecture used by runs made before QK-norm and the longer context were added."""
    c = make_cfg(size, LEGACY_T[size], False)
    del c["qkn"]
    return c


def peak_lr(n, qkn=True):
    if not qkn:
        return 2e-3 if n < 25e6 else 1.2e-3 if n < 60e6 else 6e-4 if n < 150e6 else 4e-4
    return 2e-3 if n < 25e6 else 1.5e-3 if n < 60e6 else 9e-4 if n < 150e6 else 6e-4


def lr_at(peak, step, warm, total, decay=0.2, floor=0.1):
    """Warmup-stable-decay: linear warmup, flat at peak, then a cosine decay over the last `decay` of the run."""
    if step <= warm:
        return peak * step / warm
    d0 = total * (1 - decay)
    if step <= d0:
        return peak
    p = min(1.0, (step - d0) / max(1.0, total - d0))
    return peak * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * p)))


def lr_cos(peak, step, warm, total):
    """Cosine schedule used by older runs; kept so they resume with the same curve."""
    l = peak * (0.1 + 0.45 * (1 + math.cos(math.pi * (step - 1) / max(total, 1))))
    return l * step / warm if step <= warm else l


def bench(cfg, dev, dtype, doc=False, B=8, steps=4):
    m = GPT(cfg).to(dev)
    m.use_doc_mask = doc
    amp = dev == "cuda"
    x = torch.randint(4, cfg["V"], (B, cfg["T"]), device=dev)
    best = None
    try:
        for i in range(steps):
            if dev == "cuda":
                torch.cuda.synchronize()
            t = time.time()
            with torch.autocast(dev, dtype=dtype, enabled=amp):
                loss = m(x, x)
            loss.backward()
            m.zero_grad(set_to_none=True)
            if dev == "cuda":
                torch.cuda.synchronize()
            d = time.time() - t
            if i > 0 and (best is None or d < best):
                best = d
    except torch.cuda.OutOfMemoryError:
        return None, m.nparams()
    n = m.nparams()
    del m
    if dev == "cuda":
        torch.cuda.empty_cache()
    return B * cfg["T"] / best, n


def make_plan(args, out_dir, dev, dtype):
    pp = os.path.join(out_dir, "plan.json")
    if os.path.exists(pp):
        return json.load(open(pp))
    names = ["nano"] if args.size == "nano" else ["xs", "s", "m", "l"]
    total = args.hours * 3600
    res = {}
    for nm in names:
        tps, n = bench(
            make_cfg(nm, args.ctx, args.qk_norm), dev, dtype, bool(args.doc_mask)
        )
        if tps:
            res[nm] = (tps, n)
            log(f"  {nm}: {n / 1e6:.0f}M params, {tps:.0f} tok/s")
    if args.size == "auto":
        pick = names[0]
        for nm in names:
            if nm in res and res[nm][0] * total / res[nm][1] >= args.ratio:
                pick = nm
    else:
        pick = args.size
    if pick not in res:
        raise SystemExit(f"size {pick} does not fit on this device")
    tps, n = res[pick]
    toks = tps * total
    pre_toks = tps * total * args.pre_frac
    plan = dict(
        size=pick,
        tps=tps,
        params=n,
        tokens=toks,
        pre_mb=int(min(args.max_pre_mb, max(50, pre_toks * 4.3 / 1e6 * 1.1))),
        cfg=make_cfg(pick, args.ctx, args.qk_norm),
    )
    log(
        f"plan: {pick}, {n / 1e6:.0f}M params, context {plan['cfg']['T']}, ~{tps:.0f} tok/s, "
        f"~{toks / 1e6:.0f}M tokens in {args.hours}h ({toks / n:.1f} tokens/param)"
    )
    json.dump(plan, open(pp, "w"))
    return plan


def atomic_save(obj, path):
    torch.save(obj, path + ".tmp")
    os.replace(path + ".tmp", path)


def pick_batch(model, cfg, dev, dtype, tokens_per_step):
    B = 64 if dev == "cuda" else 4
    while B >= 1:
        try:
            x = torch.randint(4, cfg["V"], (B, cfg["T"]), device=dev)
            with torch.autocast(dev, dtype=dtype, enabled=dev == "cuda"):
                model(x, x).backward()
            model.zero_grad(set_to_none=True)
            return B, max(1, tokens_per_step // (B * cfg["T"]))
        except torch.cuda.OutOfMemoryError:
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            B //= 2
        except RuntimeError:
            model.zero_grad(set_to_none=True)
            B //= 2
    raise SystemExit("model does not fit in GPU memory")


def sample(model, tok, prompt, dev, dtype, chat=True, mx=100, rng=None, **kw):
    eng = Engine(model, tok, dev, dtype)
    return eng.reply(prompt, mx, rng, **kw) if chat else eng.cont(prompt, mx, rng, **kw)


def fmt_params(n):
    return f"{n / 1e6:.0f}M" if n >= 1e6 else f"{n / 1e3:.0f}K"


class EMA:
    """Exponential moving average of the weights; the horizon grows from ~10 steps up to 1/(1-decay)."""

    def __init__(s, model, decay):
        s.p = list(model.parameters())
        s.avg = [p.detach().clone() for p in s.p]
        s.decay, s.n = decay, 0

    @torch.no_grad()
    def update(s):
        s.n += 1
        d = min(s.decay, (1 + s.n) / (10 + s.n))
        torch._foreach_mul_(s.avg, d)
        torch._foreach_add_(s.avg, [p.detach() for p in s.p], alpha=1 - d)

    def load(s, avg, n):
        s.avg = [a.to(p.device, p.dtype) for a, p in zip(avg, s.p)]
        s.n = n

    @contextmanager
    def applied(s):
        """Temporarily swap the averaged weights into the model."""
        bak = [p.detach().clone() for p in s.p]
        with torch.no_grad():
            for p, a in zip(s.p, s.avg):
                p.copy_(a)
        try:
            yield
        finally:
            with torch.no_grad():
                for p, b in zip(s.p, bak):
                    p.copy_(b)


def probe(model, tok, dev, dtype, chat, out_dir=None, name="", step=0, elapsed=0):
    """Sample the fixed prompts with a fixed seed, log a few, append all to samples.jsonl.
    Returns the share of identity questions answered with the right name (chat only)."""
    model.eval()
    rng = make_rng(7, dev)
    rows, hit, need = [], 0, 0
    for p in PROMPTS if chat else BASE_PROMPTS:
        eng = Engine(model, tok, dev, dtype)
        r = eng.reply(p, 60, rng, keep=False) if chat else eng.cont(p, 40, rng, temp=0.8)
        r = r.replace("\n", " ")
        rows.append(
            dict(stage=name, step=step, elapsed=int(elapsed), prompt=p, reply=r)
        )
        if chat and p in IDENT:
            need += 1
            hit += IDENT[p].lower() in r.lower()
    for r in rows[:3]:
        log(f"  > {r['prompt']} -> {r['reply'][:160]}")
    if out_dir:
        with open(os.path.join(out_dir, "samples.jsonl"), "a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    if chat and need:
        log(f"  identity check: {hit}/{need}")
        return hit / need
    return None


def save_model(path, state, cfg, pt=False):
    """Weights only (safetensors), so loading a shared model never runs pickled code."""
    save_st(path, state, {"cfg": json.dumps(cfg), "format": "iludev"})
    if pt:
        atomic_save({"model": state, "cfg": cfg}, os.path.splitext(path)[0] + ".pt")


def load_into(model, sd, src):
    try:
        model.load_state_dict(sd)
    except RuntimeError as e:
        raise SystemExit(
            f"{src} does not match this model ({str(e)[:200]}). It was probably made by an older "
            "version: use a fresh --out folder, or keep the old plan.json so the old architecture is used."
        )


def train_stage(
    args, out_dir, name, cfg, data, chat, hours, peak, epochs, init, tok, dev, dtype
):
    last = os.path.join(out_dir, f"last_{name}.pt")
    final = os.path.join(out_dir, f"{name}.safetensors")
    csv_path = os.path.join(out_dir, "train_log.csv")
    model = GPT(cfg).to(dev)
    model.use_doc_mask = bool(args.doc_mask)
    resumed = None
    if os.path.exists(last):
        resumed = torch.load(last, map_location=dev, weights_only=True)
        load_into(model, resumed["model"], last)
        log(f"resuming {name} from step {resumed['step']}/{resumed['total']}")
    elif init:
        load_into(model, read_ckpt(init, dev)["model"], init)
    decay = [p for p in model.parameters() if p.dim() >= 2]
    nodecay = [p for p in model.parameters() if p.dim() < 2]
    kw = dict(fused=True) if dev == "cuda" else {}
    opt = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": 0.1},
            {"params": nodecay, "weight_decay": 0.0},
        ],
        lr=peak,
        betas=(0.9, 0.95),
        **kw,
    )
    if resumed:
        opt.load_state_dict(resumed["opt"])
    scaler = torch.amp.GradScaler(
        "cuda", enabled=dev == "cuda" and dtype == torch.float16
    )
    if resumed and resumed.get("scaler"):
        scaler.load_state_dict(resumed["scaler"])
    ema = EMA(model, args.ema) if args.ema > 0 else None
    if ema and resumed and resumed.get("ema"):
        ema.load(resumed["ema"], resumed.get("ema_n", 0))
    T = cfg["T"]
    tps_target = args.tokens_per_step
    B, accum = pick_batch(model, cfg, dev, dtype, tps_target)
    per = B * T * accum
    run = model
    if dev == "cuda" and args.compile:
        try:
            run = torch.compile(model)
        except Exception as e:
            log("compile unavailable:", e)
    rng = np.random.default_rng(args.seed + (resumed["step"] if resumed else 0))
    amp = dev == "cuda"
    pf = Prefetch(data, B, rng) if dev == "cuda" else None

    def step_fn():
        tot = 0.0
        for _ in range(accum):
            x, y = pf.get() if pf else data.batch(B, rng)
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast(dev, dtype=dtype, enabled=amp):
                loss = run(x, y) / accum
            scaler.scale(loss).backward()
            tot += loss.item()
        return tot

    def measure():
        """Seconds per optimizer step (best of two timed steps after one warm-up)."""
        nonlocal scaler
        sps = float("inf")
        for i in range(3):
            if dev == "cuda":
                torch.cuda.synchronize()
            t = time.time()
            step_fn()
            if dev == "cuda":
                torch.cuda.synchronize()
            if i >= 1:
                sps = min(sps, time.time() - t)
        opt.zero_grad(set_to_none=True)
        scaler = torch.amp.GradScaler(
            "cuda", enabled=dev == "cuda" and dtype == torch.float16
        )
        return sps

    sched = resumed.get("sched", "cos") if resumed else "wsd"
    dfrac = resumed.get("decay_frac", args.decay_frac) if resumed else args.decay_frac
    epoch_steps = max(1, int(epochs * data.ntokens() / per))
    if resumed:
        total, step, elapsed = resumed["total"], resumed["step"], resumed["elapsed"]
        warm = resumed.get("warm") or min(300, max(1, total // 30))
        if resumed.get("early"):
            total = step
        elif args.extend_hours > 0 and step < total:
            if sched != "wsd" or step > (1 - dfrac) * total:
                log("  --extend-hours ignored: this stage is already decaying or came from an older version")
            else:
                total = step + max(1, int(args.extend_hours * 3600 / measure() * 0.95))
                log(f"  extended to {total} steps")
    else:
        step, elapsed = 0, 0.0
        total = max(1, min(epoch_steps, int(hours * 3600 / measure() * 0.95)))
        warm = min(300, max(1, total // 30))

    def lr_fn(s):
        if sched == "wsd":
            return lr_at(peak, s, warm, total, dfrac)
        return lr_cos(peak, s, warm, total)

    log(
        f"{name}: {total} steps of {per} tokens ({total * per / 1e6:.0f}M tokens), batch {B}x{accum}, lr {peak:.1e}"
        + (", EMA" if ema else "")
        + (", doc mask" if model.use_doc_mask else "")
    )
    if step >= total:
        if not os.path.exists(final):
            save_model(final, model.state_dict(), cfg, args.save_pt)
        return final
    stop = {"v": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(v=True))
    st = {"early": False}
    best = dict(resumed["best"]) if resumed and resumed.get("best") else dict(v=float("inf"), bad=0)

    def save_last():
        extra = {"ema": [t.cpu() for t in ema.avg], "ema_n": ema.n} if ema else {}
        atomic_save(
            {
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "scaler": scaler.state_dict(),
                "step": step,
                "total": total,
                "elapsed": elapsed,
                "warm": warm,
                "sched": sched,
                "decay_frac": dfrac,
                "early": st["early"],
                "best": best,
                "cfg": cfg,
                **extra,
            },
            last,
        )

    def val_loss():
        model.eval()
        vl = []
        r2 = np.random.default_rng(7)
        with torch.no_grad():
            for _ in range(10):
                x, y = data.batch(B, r2, val=True)
                with torch.autocast(dev, dtype=dtype, enabled=amp):
                    vl.append(model(x.to(dev), y.to(dev)).item())
        return float(np.mean(vl))

    def checkpoint():
        v_raw = val_loss()
        v_ema = None
        if ema:
            with ema.applied():
                v_ema = val_loss()
        use_ema = v_ema is not None and v_ema < v_raw
        v = v_ema if use_ema else v_raw
        if not math.isfinite(v):
            raise SystemExit(
                "val loss is not finite: training diverged. Lower --lr, or run with --compile 0."
            )
        log(
            f"  val loss {v_raw:.3f} (ppl {math.exp(min(v_raw, 20)):.1f})"
            + (f", ema {v_ema:.3f}" if ema else "")
        )
        csvlog(csv_path, name, step, "val", f"{v_raw:.4f}", "", "", int(elapsed))
        if ema:
            csvlog(csv_path, name, step, "val_ema", f"{v_ema:.4f}", "", "", int(elapsed))
        improved = v < best["v"] - 1e-4
        with ema.applied() if use_ema else nullcontext():
            if improved:
                save_model(final, model.state_dict(), cfg, args.save_pt)
            try:
                ident = probe(model, tok, dev, dtype, chat, out_dir, name, step, elapsed)
            except Exception as e:
                log("  probe failed:", repr(e))
                ident = None
        if ident is not None:
            csvlog(csv_path, name, step, "identity", f"{ident:.2f}", "", "", int(elapsed))
        if improved:
            best.update(v=v, bad=0)
        else:
            best["bad"] += 1
        save_last()
        model.train()

    model.train()
    t0, run0, lastsave = time.time(), elapsed, time.time()
    acc, n, tl, ls = 0.0, 0, time.time(), step
    ev = args.eval_every or max(50, total // 10)
    try:
        while step < total and not stop["v"]:
            step += 1
            lr = lr_fn(step)
            for g in opt.param_groups:
                g["lr"] = lr
            acc += step_fn()
            n += 1
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            if ema:
                ema.update()
            elapsed = run0 + time.time() - t0
            if step % max(1, min(50, total // 200)) == 0:
                now = time.time()
                sps = (step - ls) / (now - tl)
                left = (total - step) / sps
                log(
                    f"step {step}/{total}  loss {acc / n:.3f}  lr {lr:.1e}  {sps * per:.0f} tok/s  {int(elapsed)}s elapsed  ~{int(left)}s left"
                )
                csvlog(
                    csv_path,
                    name,
                    step,
                    "train",
                    f"{acc / n:.4f}",
                    f"{lr:.3e}",
                    int(sps * per),
                    int(elapsed),
                )
                acc, n, tl, ls = 0.0, 0, now, step
            if step < total and (
                time.time() - lastsave > args.save_min * 60 or step % ev == 0
            ):
                checkpoint()
                lastsave = time.time()
                if chat and args.patience > 0 and best["bad"] >= args.patience:
                    log(
                        f"  early stop: val loss has not improved for {best['bad']} checks, keeping the best weights"
                    )
                    st["early"] = True
                    break
    except KeyboardInterrupt:
        stop["v"] = True
    checkpoint()
    if step >= total or st["early"]:
        log(f"saved {final}")
    else:
        log("paused - run the same command again to continue")
        sys.exit(0)
    return final


def load_model(out_dir, which, dev):
    from tokenizers import Tokenizer

    path = find_model(os.path.join(out_dir, which))
    ck = read_ckpt(path, dev)
    m = GPT(ck["cfg"]).to(dev)
    load_into(m, ck["model"], path)
    m.eval()
    return m, Tokenizer.from_file(os.path.join(os.path.dirname(path), "tokenizer.json"))


def gen_opts(args):
    return dict(
        temp=args.temp,
        topk=args.topk,
        topp=args.topp,
        minp=args.minp,
        rep=args.rep,
        freq=args.freq,
    )


def cmd_chat(args):
    dev, gpu, dtype = device_info()
    m, tok = load_model(args.out, args.model, dev)
    eng = Engine(m, tok, dev, dtype)
    rng = make_rng(args.seed, dev)
    kw = gen_opts(args)
    if args.p:
        print(eng.reply(args.p, args.max, rng, **kw))
        return
    log(
        f"{NAME} by {ORG} (founder: {FOUNDER}) - {fmt_params(m.nparams())} params, context {m.cfg['T']}. Commands: /reset /clear /quit"
    )
    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            return
        cmd = line.lower()
        if cmd in ("/quit", "/exit", "quit", "exit"):
            return
        if cmd in ("/reset", "/clear", "/new", "reset", "clear"):
            eng.reset()
            eng.last_ids = []
            print("(conversation cleared)")
            continue
        if cmd in ("/help", "/?", "help"):
            print("Commands: /reset (or /clear, /new) /quit (or /exit)")
            continue
        if line:
            print("iludev>", eng.reply(line, args.max, rng, **kw))


def prep_data(args, out, cfg, mb, nano):
    data_dir = os.path.join(out, "data")
    os.makedirs(data_dir, exist_ok=True)
    if nano:
        shares = {}
    else:
        base = dict(web=0.45, cosmo=0.25, wiki=0.12, wiki_id=0.05, stories=0.13)
        k = 1 - args.code_share - args.math_share
        shares = {a: v * k for a, v in base.items()}
        shares.update(code=args.code_share, math=args.math_share)
    txt = os.path.join(data_dir, "pre.txt")
    if not args.no_download:
        collect_text(txt, mb, args.text, args.crawl_out, shares, bool(args.quality))
    else:
        collect_text(txt, 0, args.text, args.crawl_out, {}, bool(args.quality))
    rng = random.Random(args.seed)
    convs_path = os.path.join(data_dir, "chats.json")
    if os.path.exists(convs_path):
        convs = json.load(open(convs_path))
    else:
        convs = load_chat_sources(args, data_dir)
        real = len(convs)
        convs += identity(args.id_rep, rng)
        rng.shuffle(convs)
        json.dump(convs, open(convs_path, "w"))
        log(f"  total {len(convs)} conversations ({real} real)")
    ident = os.path.join(data_dir, "identity.txt")
    if not os.path.exists(ident):
        with open(ident, "w") as f:
            for c in identity(3, random.Random(1)):
                f.write("\n".join(c) + "\n\n")
    tok = build_tokenizer(out, [txt, ident], cfg["V"], bool(args.split_digits))
    pre_bin = os.path.join(data_dir, "pre.bin")
    encode_pre(tok, txt, pre_bin)
    chat_bin, chat_mask = os.path.join(data_dir, "chat.bin"), os.path.join(
        data_dir, "chat.mask"
    )
    encode_chat(tok, convs, chat_bin, chat_mask)
    return tok, pre_bin, chat_bin, chat_mask


def cmd_prep(args):
    out = args.out
    size = "m" if args.size == "auto" else args.size
    cfg = make_cfg(size, args.ctx, args.qk_norm)
    mb = args.pre_mb or 4500
    log(
        f"prep only (no GPU needed): size {size}, ~{mb} MB of pretraining text. Then run the train command on the GPU runtime."
    )
    prep_data(args, out, cfg, mb, size == "nano")
    log(
        "prep done. now run: python train.py train --out "
        + out
        + " --size "
        + size
        + " --hours <H>"
    )


def cmd_train(args):
    dev, gpu, dtype = device_info()
    log(f"device: {gpu}, dtype {str(dtype).replace('torch.', '')}")
    if dev == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    out = args.out
    os.makedirs(os.path.join(out, "data"), exist_ok=True)
    torch.manual_seed(args.seed)
    plan = make_plan(args, out, dev, dtype)
    cfg = plan.get("cfg")
    if not cfg:
        cfg = legacy_cfg(plan["size"])
        log(
            f"  this run was made by an older version: keeping its original architecture (no QK-norm, {cfg['T']}-token context)"
        )
    mb = args.pre_mb or plan["pre_mb"]
    tok, pre_bin, chat_bin, chat_mask = prep_data(
        args, out, cfg, mb, plan["size"] == "nano"
    )
    peak = args.lr or peak_lr(plan["params"], bool(cfg.get("qkn")))
    pre = Data(localize(pre_bin, args.local_data), None, cfg["T"], False)
    base = train_stage(
        args,
        out,
        "base",
        cfg,
        pre,
        False,
        args.hours * args.pre_frac,
        peak,
        args.pre_epochs,
        None,
        tok,
        dev,
        dtype,
    )
    ch = Data(
        localize(chat_bin, args.local_data),
        localize(chat_mask, args.local_data),
        cfg["T"],
        True,
    )
    train_stage(
        args,
        out,
        "chat",
        cfg,
        ch,
        True,
        args.hours * (1 - args.pre_frac),
        peak * 0.3,
        args.ft_epochs,
        base,
        tok,
        dev,
        dtype,
    )
    log(f"done. chat with: python {sys.argv[0]} chat --out {out}")


def hs_clean(t):
    t = t.strip().replace(" [title]", ". ")
    return re.sub(r"\[.*?\]", "", t).replace("  ", " ")


def eval_items(task, path=""):
    """Yield (context, [endings], correct_index) for a multiple-choice task."""
    if task == "local":
        for line in open(path, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                yield r["ctx"], r["endings"], int(r["label"])
        return
    from datasets import load_dataset

    if task == "hellaswag":
        err = None
        for name in ("Rowan/hellaswag", "hellaswag"):
            try:
                ds = load_dataset(name, split="validation", streaming=True)
                break
            except Exception as e:
                err = e
        else:
            raise err
        for r in ds:
            ctx = hs_clean(
                r["activity_label"] + ": " + r["ctx_a"] + " " + r["ctx_b"].capitalize()
            )
            yield ctx, [" " + hs_clean(e) for e in r["endings"]], int(r["label"])
    elif task == "arc_easy":
        ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test", streaming=True)
        for r in ds:
            labels = r["choices"]["label"]
            if r["answerKey"] in labels:
                yield (
                    "Question: " + r["question"] + "\nAnswer:",
                    [" " + t for t in r["choices"]["text"]],
                    labels.index(r["answerKey"]),
                )
    else:
        raise ValueError(f"unknown task {task}")


@torch.no_grad()
def score_choices(model, tok, items, dev, dtype, limit):
    """Zero-shot multiple choice by log-likelihood of each ending given the context.
    acc picks the highest total log-prob, acc_norm the highest per character."""
    T = model.cfg["T"]
    model.eval()
    ok = okn = n = 0
    chance = 0.0
    for ctx, ends, label in items:
        if n >= limit:
            break
        c = [EOD] + tok.encode(ctx).ids
        eids = [tok.encode(e).ids for e in ends]
        if not all(eids):
            continue
        seqs = [(c + e)[-T:] for e in eids]
        nes = [min(len(e), len(s) - 1) for e, s in zip(eids, seqs)]
        L = max(map(len, seqs))
        x = torch.zeros(len(seqs), L, dtype=torch.long)
        for i, s in enumerate(seqs):
            x[i, : len(s)] = torch.tensor(s)
        with torch.autocast(dev, dtype=dtype, enabled=dev == "cuda"):
            lg = model(x.to(dev))
        lps = []
        for i, (s, ne) in enumerate(zip(seqs, nes)):
            Li = len(s)
            pos = torch.arange(Li - ne - 1, Li - 1, device=dev)
            tgt = torch.tensor(s[Li - ne :], device=dev)
            lp = lg[i, pos].float().log_softmax(-1).gather(1, tgt[:, None]).sum()
            lps.append(lp.item())
        pred = int(np.argmax(lps))
        predn = int(np.argmax([l / max(1, len(e)) for l, e in zip(lps, ends)]))
        ok += pred == label
        okn += predn == label
        chance += 1 / len(ends)
        n += 1
    if not n:
        return None
    return dict(n=n, acc=ok / n, acc_norm=okn / n, chance=chance / n)


def cmd_eval(args):
    dev, gpu, dtype = device_info()
    m, tok = load_model(args.out, args.model, dev)
    stem = os.path.splitext(os.path.basename(find_model(os.path.join(args.out, args.model))))[0]
    res = dict(model=stem, params=m.nparams(), context=m.cfg["T"])
    tasks = list(args.tasks) + (["local"] if args.eval_file else [])
    log(f"== zero-shot eval of {stem} ({fmt_params(m.nparams())} params)")
    for t in tasks:
        try:
            r = score_choices(
                m, tok, eval_items(t, args.eval_file), dev, dtype, args.eval_limit
            )
        except Exception as e:
            log(f"  skipped {t}: {e}")
            continue
        if r:
            res[t] = r
            log(
                f"  {t:10s} acc {r['acc']:.3f}  acc_norm {r['acc_norm']:.3f}  (chance {r['chance']:.3f}, n={r['n']})"
            )
    chat = "base" not in stem
    ident = probe(m, tok, dev, dtype, chat, args.out, "eval", 0, 0)
    if ident is not None:
        res["identity"] = ident
    path = os.path.join(args.out, f"eval_{stem}.json")
    json.dump(res, open(path, "w"), indent=2)
    log(f"saved {path}")


def cmd_release(args):
    """Collect weights, tokenizer, config and a model card (with real numbers from this run) into one folder."""
    out = args.out
    src = find_model(os.path.join(out, args.model))
    stem = os.path.splitext(os.path.basename(src))[0]
    ck = read_ckpt(src)
    cfg = ck["cfg"]
    n = sum(v.numel() for v in ck["model"].values())
    name = args.release_name
    rel = os.path.join(out, "release", re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-"))
    os.makedirs(rel, exist_ok=True)
    save_st(
        os.path.join(rel, "model.safetensors"),
        ck["model"],
        {"cfg": json.dumps(cfg), "format": "iludev"},
    )
    shutil.copy(os.path.join(os.path.dirname(src), "tokenizer.json"), rel)
    json.dump(
        dict(
            name=name,
            organization=ORG,
            architecture=cfg,
            parameters=n,
            special_tokens=SPECIAL,
            sampling_defaults=GEN,
        ),
        open(os.path.join(rel, "config.json"), "w"),
        indent=2,
    )

    def jload(p):
        try:
            return json.load(open(p))
        except Exception:
            return None

    plan = jload(os.path.join(out, "plan.json")) or {}
    ev = jload(os.path.join(out, f"eval_{stem}.json")) or {}
    vals = {}
    try:
        for r in csv.DictReader(open(os.path.join(out, "train_log.csv"))):
            if r["kind"] == "val":
                vals[r["stage"]] = (int(r["step"]), float(r["value"]))
    except Exception:
        pass
    samples = []
    try:
        rows = [json.loads(l) for l in open(os.path.join(out, "samples.jsonl"), encoding="utf-8")]
        rows = [r for r in rows if r["stage"] == "chat"]
        if rows:
            top = max(r["step"] for r in rows)
            samples = [r for r in rows if r["step"] == top][:6]
    except Exception:
        pass
    L = [
        f"# {name}",
        "",
        f"{name} is a {fmt_params(n)}-parameter chat language model trained from scratch by {ORG} (founder: {FOUNDER}).",
        "",
        "## Model",
        "",
        "| | |",
        "|---|---|",
        "| Architecture | decoder-only transformer: RMSNorm, rotary positions, SwiGLU"
        + (", QK-norm" if cfg.get("qkn") else "")
        + ", tied embeddings |",
        f"| Parameters | {n:,} |",
        f"| Layers / width / heads | {cfg['L']} / {cfg['d']} / {cfg['H']} |",
        f"| Context | {cfg['T']} tokens |",
        f"| Vocabulary | {cfg['V']} (byte-level BPE) |",
        "",
        "## Training",
        "",
    ]
    if plan.get("tokens"):
        L.append(f"- Planned budget: ~{plan['tokens'] / 1e6:.0f}M tokens ({plan['tokens'] / n:.1f} tokens per parameter).")
    for st_, (s_, v_) in vals.items():
        L.append(f"- Last {st_} validation loss: {v_:.3f} (perplexity {math.exp(min(v_, 20)):.1f}) at step {s_}.")
    if not (plan.get("tokens") or vals):
        L.append("- Training details were not found in the output folder.")
    L += ["", "## Evaluation (zero-shot)", ""]
    tasks = [k for k in ev if isinstance(ev[k], dict)]
    if tasks:
        L += ["| Task | Accuracy | Length-normalised | Chance | Examples |", "|---|---|---|---|---|"]
        for t in tasks:
            r = ev[t]
            L.append(f"| {t} | {r['acc']:.3f} | {r['acc_norm']:.3f} | {r['chance']:.3f} | {r['n']} |")
    else:
        L.append("Not evaluated yet. Run `python train.py eval --out <dir>` and release again.")
    if "identity" in ev:
        L += ["", f"Identity prompts answered correctly: {ev['identity']:.0%}."]
    if samples:
        L += ["", "## Sample outputs", ""]
        for r in samples:
            L.append(f"- **{r['prompt']}** {r['reply']}")
    L += [
        "",
        "## Usage",
        "",
        "```",
        "python chat.py chat --model model.safetensors",
        "python export_gguf.py --model model.safetensors --out model.gguf   # for llama.cpp",
        "```",
        "",
        "## Limitations",
        "",
        "- A small model: expect mistakes in facts, arithmetic and reasoning.",
        f"- The context window is {cfg['T']} tokens, so long conversations are truncated.",
        "- Trained on web text and synthetic chats; it can repeat their biases and errors.",
        "",
        "## License",
        "",
        "Not set yet. Choose a license before publishing.",
    ]
    open(os.path.join(rel, "README.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    log(f"release folder ready: {rel}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "cmd",
        nargs="?",
        default="train",
        choices=["train", "chat", "crawl", "prep", "eval", "release"],
    )
    ap.add_argument("--out", default=default_out())
    ap.add_argument("--size", default="auto", choices=["auto"] + list(SIZES))
    ap.add_argument("--hours", type=float, default=8.0)
    ap.add_argument("--pre-frac", type=float, default=0.55)
    ap.add_argument("--ratio", type=float, default=20.0, help="target tokens per parameter for --size auto")
    ap.add_argument("--ft-epochs", type=float, default=3.0)
    ap.add_argument("--pre-epochs", type=float, default=4.0)
    ap.add_argument("--pre-mb", type=int, default=0)
    ap.add_argument("--max-pre-mb", type=int, default=6000)
    ap.add_argument("--code-share", type=float, default=0.08)
    ap.add_argument("--math-share", type=float, default=0.05)
    ap.add_argument("--quality", type=int, default=1, help="filter crawled pages for quality")
    ap.add_argument("--split-digits", type=int, default=0, help="tokenize digits one by one (new tokenizers only)")
    ap.add_argument("--max-ultrachat", type=int, default=250000)
    ap.add_argument("--max-smoltalk", type=int, default=250000)
    ap.add_argument("--id-rep", type=int, default=8)
    ap.add_argument("--max-user-chars", type=int, default=1500)
    ap.add_argument("--max-bot-chars", type=int, default=3000)
    ap.add_argument("--skip-sources", nargs="*", default=[])
    ap.add_argument("--extra-chats", default="")
    ap.add_argument("--text", nargs="*", default=[])
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--ctx", type=int, default=0, help="context length (default 1024, nano 128)")
    ap.add_argument("--qk-norm", type=int, default=1)
    ap.add_argument("--doc-mask", type=int, default=1, help="block attention across document boundaries")
    ap.add_argument("--ema", type=float, default=0.999, help="weight EMA decay, 0 = off")
    ap.add_argument("--lr", type=float, default=0.0, help="peak learning rate override")
    ap.add_argument("--decay-frac", type=float, default=0.2, help="share of steps spent decaying the LR")
    ap.add_argument("--extend-hours", type=float, default=0.0, help="when resuming before the decay phase, train this much longer")
    ap.add_argument("--patience", type=int, default=3, help="chat stage: stop after this many checks without val improvement (0 = off)")
    ap.add_argument("--eval-every", type=int, default=0, help="steps between val checks (0 = auto)")
    ap.add_argument("--save-pt", type=int, default=0, help="also write legacy .pt weights")
    ap.add_argument("--tokens-per-step", type=int, default=65536, help="batch size in tokens per optimizer step")
    ap.add_argument("--compile", type=int, default=1)
    ap.add_argument("--local-data", type=int, default=1)
    ap.add_argument("--save-min", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--crawl-seeds", nargs="*", default=[])
    ap.add_argument("--crawl-out", default="")
    ap.add_argument("--crawl-pages", type=int, default=2000)
    ap.add_argument("--crawl-delay", type=float, default=1.0)
    ap.add_argument("--model", default="chat.safetensors")
    ap.add_argument("--p", default="")
    ap.add_argument("--max", type=int, default=150)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--topk", type=int, default=0)
    ap.add_argument("--topp", type=float, default=1.0)
    ap.add_argument("--minp", type=float, default=0.05)
    ap.add_argument("--rep", type=float, default=1.0)
    ap.add_argument("--freq", type=float, default=0.3)
    ap.add_argument("--tasks", nargs="*", default=["hellaswag", "arc_easy"])
    ap.add_argument("--eval-limit", type=int, default=1000)
    ap.add_argument("--eval-file", default="", help="local .jsonl of {ctx, endings, label}")
    ap.add_argument("--release-name", default="Xeriosa 1")
    args = ap.parse_args()
    CAPS.update(user=args.max_user_chars, bot=args.max_bot_chars)
    if not args.crawl_out:
        args.crawl_out = os.path.join(args.out, "crawl.txt")
    if not args.extra_chats:
        args.extra_chats = os.path.join(args.out, "extra", "*.txt")
    if args.cmd == "crawl":
        crawl(args)
    elif args.cmd == "prep":
        cmd_prep(args)
    elif args.cmd == "chat":
        cmd_chat(args)
    elif args.cmd == "eval":
        cmd_eval(args)
    elif args.cmd == "release":
        cmd_release(args)
    else:
        cmd_train(args)


if __name__ == "__main__":
    main()
