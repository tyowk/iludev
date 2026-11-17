#!/usr/bin/env python3
import argparse
import json
import os
import struct
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
import torch.nn as nn
import torch.nn.functional as F

NAME, ORG, FOUNDER = "Iludev", "SiLuyla Development", "Tyowk"
END, EOD, USER, BOT = 0, 1, 2, 3


def device_info():
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        return "cuda", torch.bfloat16 if p.major >= 8 else torch.float16
    return "cpu", torch.float32


class RMS(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-5).to(x.dtype) * self.w


def rope(x, cos, sin):
    h = x.shape[-1] // 2
    x1, x2 = x[..., :h], x[..., h:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1)


class KV:
    def __init__(self, H, T, hd, dev, dtype):
        self.H, self.T, self.hd, self.dev, self.dtype = H, T, hd, dev, dtype
        self.k = None
        self.v = None
        self.n = 0

    def add(self, k, v):
        need = self.n + k.shape[2]
        if self.k is None or need > self.k.shape[2]:
            cap = 256 if self.k is None else 2 * self.k.shape[2]
            cap = min(self.T, max(need, cap))
            nk = torch.empty(1, self.H, cap, self.hd, device=self.dev, dtype=self.dtype)
            nv = torch.empty(1, self.H, cap, self.hd, device=self.dev, dtype=self.dtype)
            if self.k is not None:
                nk[:, :, : self.n] = self.k[:, :, : self.n]
                nv[:, :, : self.n] = self.v[:, :, : self.n]
            self.k, self.v = nk, nv
        self.k[:, :, self.n : need] = k
        self.v[:, :, self.n : need] = v
        self.n = need
        return self.k[:, :, :need], self.v[:, :, :need]


class Attn(nn.Module):
    def __init__(self, d, H, qkn=False):
        super().__init__()
        self.H = H
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.qn = RMS(d // H) if qkn else None
        self.kn = RMS(d // H) if qkn else None

    def forward(self, x, cos, sin, cache=None, mask=None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.H, C // self.H).permute(2, 0, 3, 1, 4)
        if self.qn is not None:
            q, k = self.qn(q), self.kn(k)
        q, k = rope(q, cos, sin).to(v.dtype), rope(k, cos, sin).to(v.dtype)
        if cache is None:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=mask is None)
        else:
            k, v = cache.add(k, v)
            S = k.shape[2]
            if T == 1:
                y = F.scaled_dot_product_attention(q, k, v)
            elif S == T:
                y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            else:
                m = torch.arange(S, device=x.device)[None, :] <= (S - T + torch.arange(T, device=x.device))[:, None]
                y = F.scaled_dot_product_attention(q, k, v, attn_mask=m)
        return self.o(y.transpose(1, 2).reshape(B, T, C))


class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.ff = (int(8 * d / 3) + 63) // 64 * 64
        self.up = nn.Linear(d, 2 * self.ff, bias=False)
        self.down = nn.Linear(self.ff, d, bias=False)

    def forward(self, x):
        a, b = self.up(x).chunk(2, -1)
        return self.down(F.silu(a) * b)


class Block(nn.Module):
    def __init__(self, d, H, qkn=False):
        super().__init__()
        self.n1, self.n2 = RMS(d), RMS(d)
        self.attn, self.mlp = Attn(d, H, qkn), MLP(d)

    def forward(self, x, cos, sin, cache=None, mask=None):
        x = x + self.attn(self.n1(x), cos, sin, cache, mask)
        return x + self.mlp(self.n2(x))


def doc_mask(idx):
    e = (idx == EOD).long()
    seg = e.cumsum(1) - e
    T = idx.shape[1]
    ok = torch.ones(T, T, dtype=torch.bool, device=idx.device).tril()
    return ((seg[:, :, None] == seg[:, None, :]) & ok)[:, None]


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d, L, H, T, V = cfg["d"], cfg["L"], cfg["H"], cfg["T"], cfg["V"]
        self.use_doc_mask = False
        self.emb = nn.Embedding(V, d)
        self.blocks = nn.ModuleList([Block(d, H, bool(cfg.get("qkn"))) for _ in range(L)])
        self.norm = RMS(d)
        hd = d // H
        inv = 1.0 / (10000 ** (torch.arange(0, hd, 2).float() / hd))
        fr = torch.outer(torch.arange(T).float(), inv)
        self.register_buffer("cos", fr.cos(), persistent=False)
        self.register_buffer("sin", fr.sin(), persistent=False)
        self.apply(self._init)
        for b in self.blocks:
            nn.init.normal_(b.attn.o.weight, std=0.02 / math.sqrt(2 * L))
            nn.init.normal_(b.mlp.down.weight, std=0.02 / math.sqrt(2 * L))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def nparams(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None, caches=None, pos0=0, last_only=False):
        T = idx.shape[1]
        x = self.emb(idx)
        cos, sin = self.cos[pos0 : pos0 + T], self.sin[pos0 : pos0 + T]
        mask = doc_mask(idx) if self.use_doc_mask and caches is None else None
        for i, b in enumerate(self.blocks):
            x = b(x, cos, sin, None if caches is None else caches[i], mask)
        x = self.norm(x)
        if targets is None:
            if last_only:
                x = x[:, -1:]
            return F.linear(x, self.emb.weight)
        logits = F.linear(x, self.emb.weight).float()
        return F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1), ignore_index=-100)


def tidy(s):
    s = " ".join(s.split())
    return s.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')


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
    if seed:
        rng.manual_seed(seed)
    else:
        rng.seed()
    return rng


def pick(logits, recent, temp, topk, topp, minp, rep, freq, rng, ban=()):
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
    def __init__(self, model, tok, dev, dtype):
        self.m, self.tok, self.dev, self.dtype = model, tok, dev, dtype
        self.T = model.cfg["T"]
        self.last_ids = []
        self.reset()

    def reset(self):
        c = self.m.cfg
        dt = self.dtype if self.dev == "cuda" else torch.float32
        self.caches = [KV(c["H"], self.T, c["d"] // c["H"], self.dev, dt) for _ in self.m.blocks]
        self.hist = []

    @torch.no_grad()
    def feed(self, ids):
        logits = None
        while ids:
            if len(self.hist) >= self.T:
                keep = self.hist[-self.T // 2 :]
                self.reset()
                self.feed_raw(keep)
            room = self.T - len(self.hist)
            part, ids = ids[:room], ids[room:]
            logits = self.feed_raw(part)
        return logits

    def feed_raw(self, ids):
        x = torch.tensor([ids], device=self.dev)
        with torch.autocast(self.dev, dtype=self.dtype, enabled=self.dev == "cuda"):
            lg = self.m(x, caches=self.caches, pos0=len(self.hist), last_only=True)
        self.hist += ids
        return lg[0, -1].float()

    def gen(self, logits, mx, rng, stop_end, kw):
        out, prev = [], (self.last_ids if stop_end else [])
        for i in range(mx):
            recent = torch.tensor(out[-kw["win"] :], device=self.dev) if out and kw["win"] > 0 else None
            ban = []
            if i == 0 or not stop_end:
                ban.append(END)
            if prev and len(out) < len(prev) and out == prev[: len(out)]:
                ban.append(prev[len(out)])
            c = pick(logits, recent, kw["temp"], kw["topk"], kw["topp"], kw["minp"], kw["rep"], kw["freq"], rng, ban)
            if c < 4:
                break
            out.append(c)
            logits = self.feed([c])
        return out

    def reply(self, text, mx=150, rng=None, keep=True, **kw):
        kw = {**GEN, **kw}
        mx = max(1, min(mx, self.T // 2))
        ids = [USER] + self.tok.encode(tidy(text)).ids + [END, BOT]
        if not keep:
            self.reset()
        budget = self.T - mx - 2
        if len(self.hist) + len(ids) > budget:
            h = self.hist
            cut = next((i for i in range(1, len(h)) if h[i] in (USER, BOT) and len(h) - i + len(ids) <= budget), len(h))
            keepids = h[cut:]
            self.reset()
            if keepids:
                self.feed(keepids)
        lg = self.feed(ids[-budget:])
        out = self.gen(lg, mx, rng, True, kw)
        self.feed([END])
        self.last_ids = list(out)
        return self.tok.decode(out)

    def cont(self, text, mx=100, rng=None, **kw):
        kw = {**GEN, **kw}
        self.reset()
        lg = self.feed([EOD] + self.tok.encode(text).ids)
        return text + self.tok.decode(self.gen(lg, mx, rng, False, kw))


def load_model(model_path, tokenizer_path=None, dev=None):
    from tokenizers import Tokenizer
    model_path = find_model(model_path)
    tokenizer_path = tokenizer_path or os.path.join(os.path.dirname(os.path.abspath(model_path)), "tokenizer.json")
    dev = dev or device_info()[0]
    ck = read_ckpt(model_path, dev)
    m = GPT(ck["cfg"]).to(dev)
    m.load_state_dict(ck["model"])
    m.eval()
    return m, Tokenizer.from_file(tokenizer_path)


def run_chat(args):
    dev, dtype = device_info()
    m, tok = load_model(args.model, args.tokenizer, dev)
    eng = Engine(m, tok, dev, dtype)
    rng = make_rng(args.seed, dev)
    gen = dict(mx=args.max, temp=args.temp, topk=args.topk, topp=args.topp, minp=args.minp, rep=args.rep, freq=args.freq, rng=rng)
    if args.p:
        print(eng.reply(args.p, **gen))
        return
    print(f"{NAME} by {ORG} (founder: {FOUNDER}) - {m.nparams()/1e6:.0f}M params, context {m.cfg['T']}, {dev}. Commands: /reset /quit")
    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
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
            print("Commands: /reset  /quit")
            continue
        if line:
            print("iludev>", eng.reply(line, **gen))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    ch = sub.add_parser("chat")
    ch.add_argument("--model", required=True)
    ch.add_argument("--tokenizer", default="")
    ch.add_argument("--p", default="")
    ch.add_argument("--max", type=int, default=150)
    ch.add_argument("--temp", type=float, default=0.7)
    ch.add_argument("--topk", type=int, default=0)
    ch.add_argument("--topp", type=float, default=1.0)
    ch.add_argument("--minp", type=float, default=0.05)
    ch.add_argument("--rep", type=float, default=1.0)
    ch.add_argument("--freq", type=float, default=0.3)
    ch.add_argument("--seed", type=int, default=0)
    ch.set_defaults(func=run_chat)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
