"""Structured pruning of the Wan transformer: blocks (depth), the residual stream (hidden), attention heads and FFN
channels, chosen on calibration activations of our own data and compensated by a least-squares refit.

A spec says what stays, as indices into the model it was cut from: {"blocks": [...], "hidden": [...] | None,
"heads", "cross_heads", "ffn": int | None, "bias": bool}. `shrink` cuts a transformer to it; without a calibration it
only gives the shapes (to load a pruned run's weights into).
"""

from __future__ import annotations

import torch
from torch import nn

from windinet.wan.dit.config import PruneConfig


class Gap(nn.Module):
    """A kept block followed by dropped ones: adds their mean residual (a trainable bias) to its output."""

    def __init__(self, block: nn.Module, bias: torch.Tensor):
        super().__init__()
        self.block, self.bias = block, nn.Parameter(bias)

    def forward(self, hidden: torch.Tensor, *args) -> torch.Tensor:
        return self.block(hidden, *args) + self.bias.to(hidden.dtype)


def inner(block: nn.Module) -> nn.Module:
    return block.block if isinstance(block, Gap) else block


DAMP = 0.01  # of the mean input energy: the refit's pull towards the plain cut


def _param(t: torch.Tensor, like: nn.Parameter) -> nn.Parameter:
    return nn.Parameter(t.contiguous(), requires_grad=like.requires_grad)


def cut_linear(lin: nn.Linear, out=None, inp=None, gram: torch.Tensor | None = None) -> None:
    """Keep the outputs `out` and the inputs `inp`. With `gram` (Σ x xᵀ of the inputs, a constant 1 appended), the
    kept inputs take over what the cut ones did: weight and bias refit by least squares (FLAP / SlimGPT), damped
    towards the plain cut, which keeps directions the calibration does not span (few sigmas, few gamma tokens)."""
    w, b = lin.weight, lin.bias
    if inp is not None and gram is not None:
        idx = torch.cat([inp.to(gram.device), torch.tensor([lin.in_features], device=gram.device)])
        g = gram.double()
        a, damp = g[idx][:, idx], DAMP * g.diagonal()[idx].mean()
        full = torch.cat([w, b[:, None]], 1).double().to(g.device)
        rhs = full @ g[:, idx] + damp * full[:, idx]
        new = torch.linalg.solve(a + damp * torch.eye(len(idx), device=g.device, dtype=g.dtype), rhs.T).T.to(w)
        w, b = new[:, :-1], new[:, -1]
    elif inp is not None:
        w = w[:, inp]
    if out is not None:
        w, b = w[out], b[out]
    lin.weight, lin.bias = _param(w, lin.weight), _param(b, lin.bias)
    lin.out_features, lin.in_features = w.shape


def cut_norm(norm: nn.Module, keep: torch.Tensor) -> None:
    norm.normalized_shape = (len(keep),)
    for name in ("weight", "bias"):
        if (p := getattr(norm, name, None)) is not None:
            setattr(norm, name, _param(p[keep], p))


def cut_heads(attn: nn.Module, heads: torch.Tensor, gram: torch.Tensor | None = None) -> None:
    """Keep whole heads: RoPE splits each head's width."""
    width = attn.inner_dim // attn.heads
    ch = (heads[:, None] * width + torch.arange(width)).flatten()
    for lin in (attn.to_q, attn.to_k, attn.to_v):
        cut_linear(lin, out=ch)
    cut_linear(attn.to_out[0], inp=ch, gram=gram)
    cut_norm(attn.norm_q, ch)
    cut_norm(attn.norm_k, ch)
    attn.heads, attn.inner_dim, attn.kv_inner_dim = len(heads), len(ch), len(ch)


def readers(t: nn.Module) -> dict[str, list[nn.Linear]]:
    """Every layer that reads the residual stream, by the name of the one whose input gram stands for the group."""
    groups = {"condition_embedder.time_proj": [t.condition_embedder.time_proj], "proj_out": [t.proj_out]}
    for i, block in enumerate(t.blocks):
        b, name = inner(block), f"blocks.{i}"
        groups |= {
            f"{name}.attn1.to_q": [b.attn1.to_q, b.attn1.to_k, b.attn1.to_v],
            f"{name}.attn2.to_q": [b.attn2.to_q],
            f"{name}.attn2.to_k": [b.attn2.to_k, b.attn2.to_v],
            f"{name}.ffn.net.0.proj": [b.ffn.net[0].proj],
        }
    return groups


def cut_stream(t: nn.Module, keep: torch.Tensor, grams: dict | None = None) -> None:
    """Keep the residual-stream channels `keep` in every layer that writes, normalises or reads it."""
    dim, grams = t.patch_embedding.out_channels, grams or {}
    for name, layers in readers(t).items():
        for lin in layers:
            cut_linear(lin, inp=keep, gram=grams.get(name))
    emb = t.condition_embedder
    cut_linear(emb.time_proj, out=(torch.arange(6)[:, None] * dim + keep).flatten())
    for lin in (emb.time_embedder.linear_2, emb.text_embedder.linear_2):
        cut_linear(lin, out=keep)
    conv = t.patch_embedding
    conv.weight, conv.bias = _param(conv.weight[keep], conv.weight), _param(conv.bias[keep], conv.bias)
    conv.out_channels = len(keep)
    for block in t.blocks:
        b = inner(block)
        for lin in (b.attn1.to_out[0], b.attn2.to_out[0], b.ffn.net[2]):
            cut_linear(lin, out=keep)
        for norm in (b.norm1, b.norm2, b.norm3):
            cut_norm(norm, keep)
        b.scale_shift_table = _param(b.scale_shift_table[..., keep], b.scale_shift_table)
        if isinstance(block, Gap):
            block.bias = _param(block.bias[keep], block.bias)
    cut_norm(t.norm_out, keep)
    t.scale_shift_table = _param(t.scale_shift_table[..., keep], t.scale_shift_table)


def shrink(t: nn.Module, spec: dict, scores: dict | None = None, refit: bool = True) -> None:
    """Cut `t` to `spec` in place: heads and FFN channels by `scores` (`importance`), with `refit` on its grams.
    Without `scores` only the shapes are right, for the weights of a pruned run to be loaded into."""
    grams = scores["grams"] if scores and refit else {}

    def keep(n: int, key: str, i: int) -> torch.Tensor:
        return scores[key][i].topk(n).indices.sort().values.cpu() if scores else torch.arange(n)

    for i in spec["blocks"]:
        b = inner(t.blocks[i])
        if spec["heads"]:
            cut_heads(b.attn1, keep(spec["heads"], "heads", i), grams.get(f"blocks.{i}.attn1.to_out.0"))
        if spec["cross_heads"]:
            cut_heads(b.attn2, keep(spec["cross_heads"], "cross_heads", i), grams.get(f"blocks.{i}.attn2.to_out.0"))
        if spec["ffn"]:
            ch = keep(spec["ffn"], "ffn", i)
            cut_linear(b.ffn.net[0].proj, out=ch)
            cut_linear(b.ffn.net[2], inp=ch, gram=grams.get(f"blocks.{i}.ffn.net.2"))
    hidden = torch.tensor(spec["hidden"]) if spec["hidden"] is not None else None
    if hidden is not None:
        cut_stream(t, hidden, grams)
    blocks, kept = list(t.blocks), spec["blocks"]
    for j, i in enumerate(kept):
        end = kept[j + 1] if j + 1 < len(kept) else len(blocks)
        if not spec["bias"] or end == i + 1:
            continue
        if not isinstance(blocks[i], Gap):
            dim = inner(blocks[i]).norm1.normalized_shape[0]
            blocks[i] = Gap(blocks[i], inner(blocks[i]).scale_shift_table.new_zeros(dim))
        if scores and refit:
            shift = scores["residual"][i + 1 : end].sum(0)
            blocks[i].bias.data += shift[hidden] if hidden is not None else shift
    t.blocks = nn.ModuleList(blocks[i] for i in kept)


def compose(before: dict | None, spec: dict) -> dict:
    """The spec of `spec` applied after `before`, as indices into the model `before` was cut from."""
    if before is None:
        return spec
    hidden = spec["hidden"]
    if before["hidden"] is not None:
        hidden = [before["hidden"][c] for c in hidden] if hidden is not None else before["hidden"]
    counts = {k: spec[k] or before[k] for k in ("heads", "cross_heads", "ffn")}
    blocks = [before["blocks"][i] for i in spec["blocks"]]
    return {"blocks": blocks, "hidden": hidden, "bias": spec["bias"] or before["bias"]} | counts


@torch.no_grad()
def calibrate(t: nn.Module, cfg: PruneConfig, forward, batches, scan_batches: int = 4) -> dict:
    """Activation statistics of `t` over `batches` (each is run by `forward()`): the input gram of every layer whose
    inputs a cut removes, each block's mean residual, and with cfg.blocks == 'scan' the relative velocity error of
    dropping each block alone (on the first `scan_batches`). Sums only: reduce over ranks, then `importance`."""
    names = []
    for i in range(len(t.blocks)):
        name = f"blocks.{i}"
        names += [f"{name}.attn1.to_out.0"] * bool(cfg.heads) + [f"{name}.attn2.to_out.0"] * bool(cfg.cross_heads)
        names += [f"{name}.ffn.net.2"] * bool(cfg.ffn)
    if cfg.hidden:
        names += list(readers(t))
    modules = dict(t.named_modules())
    grams, residual, hooks = {}, {}, []

    def gram_hook(name):
        def hook(module, args, out):
            x = args[0].flatten(0, -2).float()
            x = torch.cat([x, torch.ones_like(x[:, :1])], 1)
            with torch.autocast(x.device.type, enabled=False):  # bf16 sums would wreck the solves
                grams[name] = grams.get(name, 0) + x.T @ x

        return hook

    def residual_hook(i):
        def hook(module, args, out):
            r = (out.float() - args[0].float()).flatten(0, -2)
            residual[i] = residual.get(i, 0) + torch.cat([r.sum(0), r.new_tensor([len(r)])])

        return hook

    blocks, drop = t.blocks, torch.zeros(len(t.blocks), 2, device=next(t.parameters()).device)

    def attach() -> list:
        hooks = [modules[name].register_forward_hook(gram_hook(name)) for name in names]
        return hooks + [block.register_forward_hook(residual_hook(i)) for i, block in enumerate(blocks)]

    hooks = attach()
    for n, batch in enumerate(batches):
        ref = forward(batch)
        if cfg.depth and cfg.blocks == "scan" and n < scan_batches:
            for h in hooks:
                h.remove()
            for i in range(len(blocks)):
                t.blocks = nn.ModuleList(b for j, b in enumerate(blocks) if j != i)
                drop[i] += torch.stack([(forward(batch) - ref).square().sum(), ref.square().sum()])
            t.blocks = blocks
            hooks = attach()
    for h in hooks:
        h.remove()
    return {"grams": grams, "residual": torch.stack([residual[i] for i in range(len(blocks))]), "drop": drop}


def importance(t: nn.Module, cfg: PruneConfig, stats: dict) -> dict:
    """From the reduced statistics: per-block scores of heads and FFN channels (the output energy each one adds,
    E[x²] ‖W[:, c]‖² summed over its channels), stream channels (the same over every reader), the mean residual of
    each block, the block-drop error."""
    grams = stats["grams"]

    def energy(lin: nn.Linear, gram: torch.Tensor) -> torch.Tensor:
        n = lin.in_features
        return gram.diagonal()[:n] / gram[n, n] * lin.weight.float().square().sum(0)

    out = {"grams": grams, "residual": stats["residual"][:, :-1] / stats["residual"][:, -1:]}
    out["drop"] = (stats["drop"][:, 0] / stats["drop"][:, 1].clamp(min=1e-12)).tolist()
    for key, attn in (("heads", "attn1"), ("cross_heads", "attn2")):
        if getattr(cfg, key):
            out[key] = [
                energy(getattr(inner(b), attn).to_out[0], grams[f"blocks.{i}.{attn}.to_out.0"])
                .view(getattr(inner(b), attn).heads, -1)
                .sum(1)
                for i, b in enumerate(t.blocks)
            ]
    if cfg.ffn:
        out["ffn"] = [energy(inner(b).ffn.net[2], grams[f"blocks.{i}.ffn.net.2"]) for i, b in enumerate(t.blocks)]
    if cfg.hidden:
        out["hidden"] = sum(energy(lin, grams[name]) for name, layers in readers(t).items() for lin in layers)
    return out


def choose(t: nn.Module, cfg: PruneConfig, scores: dict) -> dict:
    """The spec of `cfg` on `t`; the channels and heads themselves are picked in `shrink` from the same scores."""
    n = len(t.blocks)
    blocks = list(range(n))
    if cfg.depth and cfg.blocks == "spread":
        blocks = sorted({round(x) for x in torch.linspace(0, n - 1, cfg.depth).tolist()})
    elif cfg.depth:
        middle = torch.tensor(scores["drop"][1:-1]).topk(cfg.depth - 2).indices + 1
        blocks = sorted({0, n - 1, *middle.tolist()})
    hidden = None
    if cfg.hidden:
        hidden = scores["hidden"].topk(cfg.hidden).indices.sort().values.tolist()
    bias = bool(cfg.depth and cfg.compensate)
    return {"blocks": blocks, "hidden": hidden, "heads": cfg.heads, "cross_heads": cfg.cross_heads, "ffn": cfg.ffn,
            "bias": bias}  # fmt: skip
