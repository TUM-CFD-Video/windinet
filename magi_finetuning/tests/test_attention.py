"""Check MAGI-1's attention, running on magi_finetuning.compat's SDPA stand-ins, against a naive
softmax(QK^T / sqrt(d)) V with the equivalent dense mask. Small random inputs, one GPU.

Covers DiT self-attention through MAGI's own FullyParallelAttention.core_attention (its per-chunk
loop) in both kv-range modes -- block-causal and the noise2clean window -- and the varlen
cross-attention. Forward and backward. Fused SDPA kernels only (restrict_sdpa_backends).

Errors are RELATIVE (max |difference| / max |fp32 reference|), for outputs and gradients. The pass
criterion is self-calibrating: the same naive formula computed in bf16 shows how much error any bf16
attention has (real flash-attn included); the stand-ins pass if they stay within 2x that baseline.

Run on a compute node (never the login node):
    sbatch jobs/jupiter/magi1/quick_test.sbatch -m magi_finetuning.tests.test_attention
"""

from types import SimpleNamespace

import magi_finetuning.compat as compat  # noqa: F401  -- must precede inference.*

import numpy as np
import torch

from inference.model.dit.dit_module import FullyParallelAttention

torch.manual_seed(0)
CHUNKS, TOK, HQ, HKV, D = 4, 96, 24, 8, 128  # 4 chunks of 96 tokens, MAGI-1 4.5B head layout
S = CHUNKS * TOK


def naive(q, k, v, mask, dtype=torch.float32):
    """q (S, HQ, D), k/v (Sk, HKV, D), mask (S, Sk) bool -> (S, HQ, D); computed in `dtype`, GQA by repetition."""
    q, k, v = q.to(dtype), k.to(dtype).repeat_interleave(HQ // HKV, 1), v.to(dtype).repeat_interleave(HQ // HKV, 1)
    scores = torch.einsum("qhd,khd->hqk", q, k) / D**0.5
    scores = scores.masked_fill(~mask[None], float("-inf"))
    return torch.einsum("hqk,khd->qhd", scores.softmax(-1), v)


def relative_errors(fn, inputs, reference_out, reference_grads, upstream):
    """Run fn on bf16 copies of inputs; relative error of its output and of every input gradient."""
    xs = [x.clone().requires_grad_(True) for x in inputs]
    out = fn(*xs)
    out.float().backward(upstream)
    rel = lambda a, b: ((a.float() - b).abs().max() / b.abs().max()).item()
    return rel(out, reference_out), max(rel(x.grad, g) for x, g in zip(xs, reference_grads))


def compare(name, magi_fn, ref_fn, inputs):
    """Stand-in (MAGI's code on SDPA, bf16) and naive-in-bf16 baseline, both against naive in fp32.
    Returns True if the stand-in is within 2x the bf16 baseline for outputs and gradients."""
    ref_inputs = [x.clone().float().requires_grad_(True) for x in inputs]
    ref_out = ref_fn(*ref_inputs)
    upstream = torch.randn_like(ref_out)
    ref_out.backward(upstream)
    ref_grads = [x.grad for x in ref_inputs]

    magi_fwd, magi_grad = relative_errors(magi_fn, inputs, ref_out.detach(), ref_grads, upstream)
    base_fwd, base_grad = relative_errors(lambda *x: ref_fn(*x, dtype=torch.bfloat16), inputs,
                                          ref_out.detach(), ref_grads, upstream)
    ok = magi_fwd <= 2 * base_fwd + 1e-3 and magi_grad <= 2 * base_grad + 1e-3
    print(f"{name:34s} relative error  output {magi_fwd:.2e} (bf16 baseline {base_fwd:.2e})   "
          f"grads {magi_grad:.2e} (baseline {base_grad:.2e})   {'ok' if ok else 'TOO LARGE'}")
    return ok


def self_attention(k_chunks):
    """k_chunks[i] = (first, last+1) chunk that chunk i attends to."""
    q_range = np.array([[i * TOK, (i + 1) * TOK] for i in range(CHUNKS)])
    k_range = np.array([[s * TOK, e * TOK] for s, e in k_chunks])
    meta = SimpleNamespace(
        denoising_range_num=CHUNKS,
        core_attn_params=SimpleNamespace(np_q_range=q_range, np_k_range=k_range),
    )
    mask = torch.zeros(S, S, dtype=torch.bool, device="cuda")
    for (qs, qe), (ks, ke) in zip(q_range, k_range):
        mask[qs:qe, ks:ke] = True

    def magi(q, k, v):  # MAGI's own loop; inputs in its "(sq b) hn hd" layout with b = 1
        return FullyParallelAttention.core_attention(None, q, k, v, bs=1, meta_args=meta)

    return magi, lambda q, k, v, dtype=torch.float32: naive(q, k, v, mask, dtype)


def main():
    compat.restrict_sdpa_backends()
    qkv = [torch.randn(S, h, D, device="cuda", dtype=torch.bfloat16) for h in (HQ, HKV, HKV)]
    results = []

    # Training pattern: chunk i sees chunks 0..i (MAGI's generate_default_kvrange).
    results.append(compare("self-attn  block-causal", *self_attention([(0, i + 1) for i in range(CHUNKS)]), qkv))
    # noise2clean window: chunk 0 clean (W=1), then W = 5, 3, 2 -> k_start = max(0, end - W).
    windows = [1, 5, 3, 2]
    results.append(compare("self-attn  noise2clean [1,5,3,2]",
                           *self_attention([(max(0, i + 1 - w), i + 1) for i, w in enumerate(windows)]), qkv))

    # Cross-attention: every chunk's queries attend to its own L caption tokens (varlen).
    L = 4
    cu_q = torch.tensor([i * TOK for i in range(CHUNKS + 1)], device="cuda", dtype=torch.int32)
    cu_k = torch.tensor([i * L for i in range(CHUNKS + 1)], device="cuda", dtype=torch.int32)
    xq = torch.randn(S, HQ, D, device="cuda", dtype=torch.bfloat16)
    xk, xv = (torch.randn(CHUNKS * L, HKV, D, device="cuda", dtype=torch.bfloat16) for _ in range(2))
    xmask = torch.zeros(S, CHUNKS * L, dtype=torch.bool, device="cuda")
    for i in range(CHUNKS):
        xmask[i * TOK:(i + 1) * TOK, i * L:(i + 1) * L] = True
    results.append(compare(
        "cross-attn varlen",
        lambda q, k, v: compat.flash_attn_varlen_func(q, k, v, cu_q, cu_k, TOK, L),
        lambda q, k, v, dtype=torch.float32: naive(q, k, v, xmask, dtype),
        [xq, xk, xv],
    ))

    print("PASS: as accurate as plain bf16 attention" if all(results)
          else "FAIL: stand-in error exceeds 2x the bf16 baseline")


if __name__ == "__main__":
    main()
