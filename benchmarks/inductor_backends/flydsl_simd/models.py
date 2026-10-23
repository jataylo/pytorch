"""Model zoo for bench_models.py: small versions of common architectures.

Each builder returns a zero-argument callable that runs one step (training steps include
the backward pass and the optimizer). Matmuls, convolutions and attention are library
calls under either backend; what differs is every pointwise, reduction, scan and
foreach kernel around them.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


DEV = "cuda"


class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.fc1, self.fc2 = nn.Linear(d, 4 * d), nn.Linear(4 * d, d)
        self.h = h

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = (
            self.qkv(self.ln1(x))
            .view(B, T, 3, self.h, D // self.h)
            .permute(2, 0, 3, 1, 4)
        )
        att = (q @ k.transpose(-1, -2)) * (D // self.h) ** -0.5
        att = att.masked_fill(
            torch.ones(T, T, device=x.device, dtype=torch.bool).triu(1), float("-inf")
        )
        y = (att.softmax(-1) @ v).transpose(1, 2).reshape(B, T, D)
        x = x + self.proj(y)
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x)), approximate="tanh"))


class GPT(nn.Module):
    def __init__(self, vocab=32000, d=768, h=12, layers=4):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.blocks = nn.Sequential(*[Block(d, h) for _ in range(layers)])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)

    def forward(self, idx):
        return self.head(self.ln(self.blocks(self.emb(idx))))


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w, self.eps = nn.Parameter(torch.ones(d)), eps

    def forward(self, x):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(
            x
        ) * self.w


def rope(x, cos, sin):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return x * cos + torch.cat((-x2, x1), -1) * sin


class LlamaBlock(nn.Module):
    def __init__(self, d, h, kv, ff):
        super().__init__()
        self.h, self.kv, self.hd = h, kv, d // h
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, kv * self.hd, bias=False)
        self.v = nn.Linear(d, kv * self.hd, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.gate = nn.Linear(d, ff, bias=False)
        self.up = nn.Linear(d, ff, bias=False)
        self.down = nn.Linear(ff, d, bias=False)

    def forward(self, x, cos, sin):
        B, T, D = x.shape
        y = self.n1(x)
        q = self.q(y).view(B, T, self.h, self.hd).transpose(1, 2)
        k = self.k(y).view(B, T, self.kv, self.hd).transpose(1, 2)
        v = self.v(y).view(B, T, self.kv, self.hd).transpose(1, 2)
        q, k = rope(q, cos, sin), rope(k, cos, sin)
        rep = self.h // self.kv
        k, v = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.o(a.transpose(1, 2).reshape(B, T, D))
        y = self.n2(x)
        return x + self.down(F.silu(self.gate(y)) * self.up(y))


class Llama(nn.Module):
    def __init__(self, vocab=32000, d=1024, h=16, kv=4, ff=2816, layers=4, T=1024):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList([LlamaBlock(d, h, kv, ff) for _ in range(layers)])
        self.norm = RMSNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        hd = d // h
        inv = 1.0 / (10000 ** (torch.arange(0, hd, 2).float() / hd))
        f = torch.outer(torch.arange(T).float(), inv)
        f = torch.cat((f, f), -1)
        self.register_buffer("cos", f.cos(), persistent=False)
        self.register_buffer("sin", f.sin(), persistent=False)

    def forward(self, idx):
        T = idx.shape[1]
        cos, sin = (
            self.cos[:T].to(self.emb.weight.dtype),
            self.sin[:T].to(self.emb.weight.dtype),
        )
        x = self.emb(idx)
        for b in self.blocks:
            x = b(x, cos, sin)
        return self.head(self.norm(x))


class BertLayer(nn.Module):
    def __init__(self, d, h, p):
        super().__init__()
        self.h, self.p = h, p
        self.qkv, self.o = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.fc1, self.fc2 = nn.Linear(d, 4 * d), nn.Linear(4 * d, d)

    def forward(self, x, bias):
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        att = (q @ k.transpose(-1, -2)) * (D // self.h) ** -0.5 + bias
        att = F.dropout(att.softmax(-1), self.p, self.training)
        y = (att @ v).transpose(1, 2).reshape(B, T, D)
        x = self.ln1(x + F.dropout(self.o(y), self.p, self.training))
        return self.ln2(
            x + F.dropout(self.fc2(F.gelu(self.fc1(x))), self.p, self.training)
        )


class Bert(nn.Module):
    def __init__(self, vocab=30522, d=768, h=12, layers=4, T=512, p=0.1):
        super().__init__()
        self.tok, self.pos = nn.Embedding(vocab, d), nn.Embedding(T, d)
        self.ln = nn.LayerNorm(d)
        self.layers = nn.ModuleList([BertLayer(d, h, p) for _ in range(layers)])
        self.head = nn.Linear(d, vocab)
        self.p = p

    def forward(self, idx, mask):
        x = self.tok(idx) + self.pos(torch.arange(idx.shape[1], device=idx.device))
        x = F.dropout(self.ln(x), self.p, self.training)
        bias = (1.0 - mask[:, None, None, :].to(x.dtype)) * -10000.0
        for layer in self.layers:
            x = layer(x, bias)
        return self.head(x)


class ViT(nn.Module):
    def __init__(self, img=224, patch=16, d=768, h=12, layers=4, classes=1000):
        super().__init__()
        self.patch = nn.Conv2d(3, d, patch, patch)
        n = (img // patch) ** 2
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.pos = nn.Parameter(torch.randn(1, n + 1, d) * 0.02)
        self.blocks = nn.ModuleList([ViTBlock(d, h) for _ in range(layers)])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, classes)

    def forward(self, x):
        x = self.patch(x).flatten(2).transpose(1, 2)
        x = torch.cat((self.cls.expand(x.shape[0], -1, -1), x), 1) + self.pos
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln(x[:, 0]))


class ViTBlock(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h = h
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.fc1, self.fc2 = nn.Linear(d, 4 * d), nn.Linear(4 * d, d)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = (
            self.qkv(self.ln1(x))
            .view(B, T, 3, self.h, D // self.h)
            .permute(2, 0, 3, 1, 4)
        )
        y = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, T, D)
        x = x + self.proj(y)
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x))))


class MoE(nn.Module):
    """Top-2 token-choice MoE layer: router softmax/topk, per-expert dispatch via a
    sort + cumsum of expert counts, and index_add combine."""

    def __init__(self, d=1024, ff=2048, experts=8, k=2):
        super().__init__()
        self.router = nn.Linear(d, experts, bias=False)
        self.w1 = nn.Parameter(torch.randn(experts, d, ff) * d**-0.5)
        self.w2 = nn.Parameter(torch.randn(experts, ff, d) * ff**-0.5)
        self.norm = RMSNorm(d)
        self.e, self.k = experts, k

    def forward(self, x):
        B, T, D = x.shape
        h = self.norm(x).view(-1, D)
        probs = self.router(h).float().softmax(-1)
        w, idx = probs.topk(self.k, -1)
        w = (w / w.sum(-1, keepdim=True)).to(x.dtype)
        flat = idx.view(-1)
        order = flat.argsort()
        counts = torch.bincount(flat, minlength=self.e)
        offsets = torch.cumsum(counts, 0) - counts
        tok = order // self.k
        # Static-shape stand-in for the ragged grouped GEMM: every expert sees every
        # routed slot, weighted by a one-hot mask built from the cumsum offsets.
        slot = torch.arange(flat.numel(), device=x.device)
        owner = (slot[:, None] >= offsets[None, :]).sum(-1) - 1
        xs = h[tok]
        y = torch.zeros_like(xs)
        for e in range(self.e):
            m = (owner == e).to(x.dtype)[:, None]
            y = y + m * (F.silu(xs @ self.w1[e]) @ self.w2[e])
        y = y * w.view(-1)[order][:, None]
        out = torch.zeros_like(h).index_add(0, tok, y)
        return x + out.view(B, T, D)


def _train(model, loss_fn, opt="adamw"):
    params = model.parameters()
    o = (
        torch.optim.AdamW(params, lr=1e-4, foreach=True)
        if opt == "adamw"
        else torch.optim.SGD(params, lr=1e-2, foreach=True)
    )

    def step():
        loss = loss_fn()
        loss.backward()
        o.step()
        o.zero_grad(set_to_none=True)
        return loss

    return step


def gpt_train_bf16():
    m = GPT().to(DEV).to(torch.bfloat16)
    x = torch.randint(0, 32000, (8, 512), device=DEV)
    t = torch.randint(0, 32000, (8, 512), device=DEV)
    return _train(m, lambda: F.cross_entropy(m(x).float().view(-1, 32000), t.view(-1)))


def gpt_infer_bf16():
    m = GPT().to(DEV).to(torch.bfloat16).eval()
    x = torch.randint(0, 32000, (16, 512), device=DEV)
    return torch.no_grad()(lambda: m(x))


def llama_train_bf16():
    m = Llama().to(DEV).to(torch.bfloat16)
    x = torch.randint(0, 32000, (4, 1024), device=DEV)
    t = torch.randint(0, 32000, (4, 1024), device=DEV)
    return _train(m, lambda: F.cross_entropy(m(x).float().view(-1, 32000), t.view(-1)))


def llama_infer_bf16():
    m = Llama().to(DEV).to(torch.bfloat16).eval()
    x = torch.randint(0, 32000, (8, 1024), device=DEV)
    return torch.no_grad()(lambda: m(x))


def bert_train_f16():
    m = Bert().to(DEV).to(torch.float16)
    x = torch.randint(0, 30522, (16, 512), device=DEV)
    mask = torch.ones(16, 512, device=DEV, dtype=torch.long)
    mask[:, 400:] = 0
    t = torch.randint(0, 30522, (16, 512), device=DEV)
    return _train(
        m, lambda: F.cross_entropy(m(x, mask).float().view(-1, 30522), t.view(-1))
    )


def vit_infer_f16():
    m = ViT().to(DEV).to(torch.float16).eval()
    x = torch.randn(64, 3, 224, 224, device=DEV, dtype=torch.float16)
    return torch.no_grad()(lambda: m(x))


def vit_train_bf16():
    m = ViT().to(DEV).to(torch.bfloat16)
    x = torch.randn(32, 3, 224, 224, device=DEV, dtype=torch.bfloat16)
    t = torch.randint(0, 1000, (32,), device=DEV)
    return _train(m, lambda: F.cross_entropy(m(x).float(), t))


def moe_train_bf16():
    m = MoE().to(DEV).to(torch.bfloat16)
    x = torch.randn(4, 512, 1024, device=DEV, dtype=torch.bfloat16)
    return _train(m, lambda: m(x).float().pow(2).mean())


class ResNetish(nn.Module):
    def __init__(self, c=64, blocks=4):
        super().__init__()
        self.stem = nn.Conv2d(3, c, 3, padding=1)
        self.layers = nn.ModuleList()
        for _ in range(blocks):
            self.layers.append(
                nn.ModuleList(
                    [
                        nn.Conv2d(c, c, 3, padding=1),
                        nn.BatchNorm2d(c),
                        nn.Conv2d(c, c, 3, padding=1),
                        nn.BatchNorm2d(c),
                    ]
                )
            )
        self.fc = nn.Linear(c, 10)

    def forward(self, x):
        x = F.relu(self.stem(x))
        for c1, b1, c2, b2 in self.layers:
            x = F.relu(x + b2(c2(F.relu(b1(c1(x))))))
        return self.fc(x.mean((2, 3)))


def resnet_train():
    m = ResNetish().to(DEV).to(memory_format=torch.channels_last)
    x = torch.randn(64, 3, 64, 64, device=DEV).to(memory_format=torch.channels_last)
    t = torch.randint(0, 10, (64,), device=DEV)
    return _train(m, lambda: F.cross_entropy(m(x), t), opt="sgd")


def resnet_nchw_train():
    m = ResNetish().to(DEV)
    x = torch.randn(64, 3, 64, 64, device=DEV)
    t = torch.randint(0, 10, (64,), device=DEV)
    return _train(m, lambda: F.cross_entropy(m(x), t), opt="sgd")


MODELS = {
    f.__name__: f
    for f in (
        gpt_train_bf16,
        gpt_infer_bf16,
        llama_train_bf16,
        llama_infer_bf16,
        bert_train_f16,
        vit_infer_f16,
        vit_train_bf16,
        moe_train_bf16,
        resnet_train,
        resnet_nchw_train,
    )
}
