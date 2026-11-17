"""Wan 2.1 VAE with normalised latents, frozen unless `train` says otherwise.

    rgb [B, 3, F, H, W] in (-1, 1), F = 4k + 1   --encode-->   z [B, 16, (F-1)/4 + 1, H/8, W/8]   (16: latent_channels)

`encode` standardises with the checkpoint's own latents_mean / latents_std (what the Wan pipeline feeds its
transformer, so natural-video latents look like N(0, 1) per channel); `decode` undoes that.
"""

from __future__ import annotations

import math

import torch
from diffusers import AutoencoderKLWan
from torch import nn
from torch.nn.functional import pad
from torch.utils.checkpoint import checkpoint

from windinet.wan import REPO
from windinet.wan.dit.model import autocast

TEMPORAL = 4  # frames per latent frame


class WanVAE:
    """`train`: 'none' (frozen), 'io' (encoder.conv_in and decoder.conv_out, the field interface), 'decoder'
    (decoder + post_quant_conv, latents unchanged) or 'all'. `channels` > 3: the VAE reads and writes that many
    channels (`widen_io`); `latent_channels` > 16: that many latent channels (`widen_latents`), the widened layers
    in the io group. `checkpointing`:
    recompute the blocks in backward (memory for time). `precision` 'mixed': bf16 autocast inside encode and decode
    when gradients flow, fp32 weights and outputs."""

    def __init__(
        self,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.float32,
        repo: str = REPO,
        train: str = "none",
        channels: int = 3,
        latent_channels: int = 16,
        checkpointing: bool = False,
        precision: str = "fp32",
    ):
        # fp32 on purpose: the Wan pipeline keeps its VAE in fp32
        vae = AutoencoderKLWan.from_pretrained(repo, subfolder="vae", torch_dtype=dtype)
        self.copies = latent_channels // vae.config.z_dim  # of Wan's 16 channels; the config keeps its 16 statistics
        if channels > 3:
            widen_io(vae, channels)
        widened = ()
        if self.copies > 1:
            widen_latents(vae, latent_channels)
            widened = (vae.encoder.conv_out, vae.quant_conv, vae.post_quant_conv, vae.decoder.conv_in)
        self.vae = vae.to(device).eval().requires_grad_(False)
        io = (vae.encoder.conv_in, vae.decoder.conv_out)  # the field interface: its own lr in the VAE stage
        trainable = {"none": [], "io": io, "decoder": [vae.decoder, vae.post_quant_conv], "all": [vae]}[train]
        for module in trainable:
            module.requires_grad_(True)
        self.trainable_params = [p for p in vae.parameters() if p.requires_grad]
        # widened latent layers too: their new weights start at zero or as copies and need the higher lr
        self.io_params = [p for m in (*io, *widened) for p in m.parameters() if p.requires_grad]
        if precision == "mixed":  # bf16 rounding of the output, amplified by the adapter's atanh, wrecks conv_out
            for conv in io:
                conv.forward = _fp32(conv.forward)
        if checkpointing:  # recompute each block in backward: a clip's graph is several times the memory
            for block in [*vae.encoder.down_blocks, *vae.decoder.up_blocks]:
                block.forward = _checkpointed(block.forward)
        self.dtype = dtype
        self.precision, self.device = precision, torch.device(device)
        shape = (1, -1, 1, 1, 1)
        self.latents_mean, self.latents_std = (
            torch.tensor(v, device=device, dtype=dtype).repeat(self.copies).view(shape)
            for v in (vae.config.latents_mean, vae.config.latents_std)
        )

    def _autocast(self) -> torch.autocast:
        """bf16 only where gradients flow: evaluation stays fp32, comparable across runs."""
        return autocast(self.precision if torch.is_grad_enabled() else "fp32", self.device)

    @staticmethod
    def pad_frames(x: torch.Tensor) -> torch.Tensor:
        """Repeat the last frame until F = 4k + 1, the frame counts the causal encoder reads (1 + 4 + 4 ...)."""
        f = x.shape[2]
        target = math.ceil((f - 1) / TEMPORAL) * TEMPORAL + 1
        return x if target == f else torch.cat([x, x[:, :, -1:].expand(-1, -1, target - f, -1, -1)], dim=2)

    def encode(self, rgb: torch.Tensor) -> torch.Tensor:
        """Deterministic (posterior mean), normalised latents. Differentiable w.r.t. `rgb`."""
        return self.posterior(rgb)[0]

    def posterior(self, rgb: torch.Tensor, sample: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalised latents (posterior mean, or a draw with `sample`) and KL(q(z|x) || N(0, I)) of the raw
        posterior, mean per latent element."""
        with self._autocast():
            dist = self.vae.encode(self.pad_frames(rgb).to(self.dtype)).latent_dist
        z = dist.sample() if sample else dist.mode()
        mean, logvar = dist.mean.float(), dist.logvar.float()
        kl = 0.5 * (mean.square() + logvar.exp() - 1 - logvar).mean()
        return (z.to(self.dtype) - self.latents_mean) / self.latents_std, kl

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        with self._autocast():
            rgb = self.vae.decode(z * self.latents_std + self.latents_mean).sample
        return rgb.to(self.dtype)  # fp32 for the adapter's atanh near +-1


def widen_io(vae: AutoencoderKLWan, channels: int) -> None:
    """Zero-initialised extra input slots of encoder.conv_in and output slots of decoder.conv_out: on the first three
    channels the widened VAE is the pretrained one, the new channels start ignored and decoded as zero."""
    for conv, dim in ((vae.encoder.conv_in, 1), (vae.decoder.conv_out, 0)):
        extra = channels - conv.weight.shape[dim]
        conv.weight = nn.Parameter(pad(conv.weight, [0, 0] * (conv.weight.ndim - 1 - dim) + [0, extra]))
        if dim == 0:
            conv.bias = nn.Parameter(pad(conv.bias, (0, extra)))
        conv.in_channels, conv.out_channels = conv.weight.shape[1], conv.weight.shape[0]


def widen_latents(vae: AutoencoderKLWan, channels: int) -> None:
    """`channels` latent channels, a multiple of z_dim: every new channel copies an old one in the encoder (mean and
    logvar rows of encoder.conv_out, quant_conv and post_quant_conv block-diagonal per copy) and starts ignored by the
    decoder (zero columns of decoder.conv_in), so the round trip is the pretrained one."""
    n = vae.config.z_dim
    copies = channels // n
    means = torch.arange(n).repeat(copies)
    posterior = torch.cat([means, means + n])  # chunk(2): means, then logvars
    _copy_rows(vae.encoder.conv_out, posterior)
    for conv, index in ((vae.quant_conv, posterior), (vae.post_quant_conv, means)):
        _copy_rows(conv, index)
        own = torch.arange(len(index)) // n % copies  # which copy each channel belongs to
        block = (own[:, None] == own[None, :]).to(conv.weight)[..., None, None, None]
        conv.weight = nn.Parameter(conv.weight.data[:, index] * block)
    conv = vae.decoder.conv_in
    conv.weight = nn.Parameter(pad(conv.weight, [0, 0] * 3 + [0, channels - n]))
    for c in (vae.encoder.conv_out, vae.quant_conv, vae.post_quant_conv, conv):
        c.in_channels, c.out_channels = c.weight.shape[1], c.weight.shape[0]


def _copy_rows(conv: nn.Conv3d, index: torch.Tensor) -> None:
    """The output channels of `conv` become its channels `index`."""
    conv.weight, conv.bias = nn.Parameter(conv.weight.data[index]), nn.Parameter(conv.bias.data[index])


def _fp32(forward):
    """Run a module's forward in fp32 under an active autocast."""

    def run(*args, **kw):
        args = [a.float() if torch.is_tensor(a) else a for a in args]
        kw = {k: v.float() if torch.is_tensor(v) else v for k, v in kw.items()}
        with torch.autocast(args[0].device.type, enabled=False):
            return forward(*args, **kw)

    return run


def _checkpointed(block_forward):
    """Checkpoint a Wan block that reads and advances a per-pass frame cache: the forward pass runs on the real
    cache, the recompute in backward on a scratch copy of the entries this block consumed, so the caches of
    the whole pass are not kept alive until backward."""

    def forward(x, feat_cache=None, feat_idx=[0], **kw):  # noqa: B006 (the diffusers block signature)
        if feat_cache is None or not torch.is_grad_enabled():
            return block_forward(x, feat_cache, feat_idx, **kw)
        ctx = {"cache": feat_cache, "start": feat_idx[0]}

        def run(x):
            if "consumed" not in ctx:  # forward pass
                return block_forward(x, ctx["cache"], feat_idx, **kw)
            scratch = [None] * ctx["start"] + ctx["consumed"]  # recompute: the same entries at the same indices
            return block_forward(x, scratch, [ctx["start"]], **kw)

        before = list(feat_cache)
        out = checkpoint(run, x, use_reentrant=False)
        ctx["consumed"], ctx["cache"] = before[ctx["start"] : feat_idx[0]], None  # drop the live cache reference
        return out

    return forward
