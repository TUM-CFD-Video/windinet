"""Make the official MAGI-1 code (IG_work/MAGI-1) importable and trainable in the windinet venv.

Import this module before anything from ``inference.*``. It:
  1. puts IG_work/MAGI-1 and IG_work/pydeps (einops) on sys.path;
  2. registers stand-ins for flash_attn / flashinfer / timm built on torch's SDPA
     (which runs FlashAttention-2 kernels on Hopper itself);
  3. patches the two things that stop MAGI-1 from training: the @torch.no_grad
     decorators on the DiT forward path, and the Triton gating kernel, which has
     no backward.
"""

import contextlib
import importlib.machinery
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

IG_WORK = Path(__file__).resolve().parents[2]
for p in (IG_WORK / "MAGI-1", IG_WORK / "pydeps"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


# ---------------- stand-ins (flash_attn layout: (batch, seq, heads, dim)) ----------------
# Every MAGI-1 attention call (DiT self/cross, VAE) ends up in flash_attn_func below.
_FUSED_BACKENDS = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION]
_restrict_backends = False


def restrict_sdpa_backends() -> None:
    """Opt in to fused SDPA kernels only: a call no fused kernel supports raises instead of
    silently running the slow math kernel. Prints which fused backends work, once."""
    global _restrict_backends
    _restrict_backends = True
    q = torch.randn(1, 24, 256, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 8, 256, 128, device="cuda", dtype=torch.bfloat16)
    ok = []
    for be in _FUSED_BACKENDS:
        try:
            with sdpa_kernel([be]):
                F.scaled_dot_product_attention(q, k, k, enable_gqa=True)
            ok.append(be.name)
        except RuntimeError:
            pass
    if not ok:
        raise RuntimeError("no fused SDPA backend supports MAGI-1's attention shapes on this GPU")
    print(f"[compat] SDPA restricted to fused kernels; usable here: {ok} (torch picks among these)", flush=True)


def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, **_):
    assert not causal, "flash_attn aligns causal masks bottom-right, SDPA top-left; MAGI-1 never uses causal=True"
    if q.dtype not in (torch.bfloat16, torch.float16) or not (q.dtype == k.dtype == v.dtype):
        # flash-attn only takes matching half precision. Under an outer bf16 autocast with fp32
        # weights, MAGI's cross-attention LayerNorms return fp32 while V is bf16.
        q, k, v = q.bfloat16(), k.bfloat16(), v.bfloat16()
    with sdpa_kernel(_FUSED_BACKENDS) if _restrict_backends else contextlib.nullcontext():
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            dropout_p=dropout_p, scale=softmax_scale, enable_gqa=True,
        )
    return out.transpose(1, 2)


def flash_attn_qkvpacked_func(qkv, dropout_p=0.0, softmax_scale=None, causal=False, **_):
    q, k, v = qkv.unbind(dim=2)
    return flash_attn_func(q, k, v, dropout_p, softmax_scale, causal)


def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q=None, max_seqlen_k=None,
                           dropout_p=0.0, softmax_scale=None, causal=False, **_):
    cq, ck = cu_seqlens_q.tolist(), cu_seqlens_k.tolist()
    outs = [
        flash_attn_func(q[None, cq[i]:cq[i + 1]], k[None, ck[i]:ck[i + 1]], v[None, ck[i]:ck[i + 1]],
                        dropout_p, softmax_scale, causal)[0]
        for i in range(len(cq) - 1)
    ]
    return torch.cat(outs, dim=0)


def apply_rotary_emb(x, cos, sin, interleaved=False, **_):
    assert not interleaved, "only the non-interleaved rotary MAGI-1 uses is implemented"
    r = cos.shape[-1]
    x1, x2 = x[..., :r], x[..., r:2 * r]
    cos, sin = cos[None, :, None, :], sin[None, :, None, :]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos, x[..., 2 * r:]], dim=-1)


def _silu_and_mul(x):
    a, b = x.chunk(2, dim=-1)
    return F.silu(a) * b


def _bmm_fp8(*_, **__):
    raise NotImplementedError("fp8 MAGI-1 weights are not supported without flashinfer")


def _module(name, **attrs):
    mod = types.ModuleType(name)
    mod.__spec__ = importlib.machinery.ModuleSpec(name, None)  # diffusers probes deps with find_spec
    mod.__dict__.update(attrs)
    sys.modules[name] = mod
    return mod


if "flash_attn" not in sys.modules:
    _module("flash_attn", flash_attn_func=flash_attn_func, flash_attn_varlen_func=flash_attn_varlen_func,
            flash_attn_qkvpacked_func=flash_attn_qkvpacked_func)
    _module("flash_attn.flash_attn_interface", flash_attn_func=flash_attn_func)
    _module("flash_attn.layers")
    _module("flash_attn.layers.rotary", apply_rotary_emb=apply_rotary_emb)
if "flashinfer" not in sys.modules:
    _module("flashinfer", activation=_module("flashinfer.activation", silu_and_mul=_silu_and_mul),
            gemm=_module("flashinfer.gemm", bmm_fp8=_bmm_fp8))
if "timm" not in sys.modules:
    _module("timm")
    _module("timm.models")
    _module("timm.models.layers", to_2tuple=lambda x: tuple(x) if isinstance(x, (tuple, list)) else (x, x),
            trunc_normal_=torch.nn.init.trunc_normal_)


# ---------------- training patches ----------------
from inference.model.dit import dit_model, dit_module  # noqa: E402


def _range_mod(x, c_mapping, gatings):
    """Autograd-friendly range_mod_triton: row r of x (s, b, h) times gatings.flatten(0, 1)[c_mapping[r]]."""
    return x * gatings.flatten(0, 1)[c_mapping]


dit_module.range_mod_triton = _range_mod

for cls, names in (
    (dit_model.VideoDiTModel, ("get_embedding_and_meta", "forward_pre_process", "forward_post_process", "forward")),
    (dit_module.TransformerBlock, ("forward",)),
):
    for name in names:
        fn = getattr(cls, name)
        setattr(cls, name, getattr(fn, "__wrapped__", fn))  # strip @torch.no_grad()


# ---------------- inference attention pattern ----------------
def set_inference_kv_range(magi_config, mode: str = "block_causal") -> None:
    """Pick the cross-chunk attention MAGI-1's sampler uses.

    'block_causal' (default, matches how we train): every chunk sees all earlier chunks
        (MAGI's generate_default_kvrange, selected by an empty noise2clean_kvrange).
    'noise2clean': MAGI's shipped [5, 4, 3, 2] noise-dependent lookback window; ablation only.
    """
    rc = magi_config.runtime_config
    if mode == "block_causal":
        rc.noise2clean_kvrange = []
    elif mode == "noise2clean":
        rc.noise2clean_kvrange = [5, 4, 3, 2]
        rc.clean_chunk_kvrange = 1
    else:
        raise ValueError(f"unknown kv-range mode {mode!r}")
