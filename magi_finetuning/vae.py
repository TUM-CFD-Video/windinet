"""MAGI-1's VAE with 4 physical channels in and out (density, momentum_x, momentum_y, pressure).

The released VAE is RGB. Only two layers ever see the channel count:
    encoder.patch_embed.proj   Conv3d(3 -> 1024, kernel (4, 8, 8))   reads the input video
    decoder.last_layer         Conv3d(4 -> 3,    kernel 3)           writes the output video
Two ways to give it 4 channels:
    grown    (train_vae --train full / io): both layers widened from 3 to 4 channels (load_vae_4ch).
    adapter  (train_vae --train adapter): the stock RGB VAE untouched, with windinet's small 1x1
             adapters around it: 4 fields -> InAdapter -> 3 -> encoder ... decoder -> 3 -> OutAdapter -> 4
             fields (load_vae_with_adapters). The encoder stays frozen, so its latent space is MAGI's own.
Checkpoints say which in their metadata ("vae_mode"); load_trained_vae() rebuilds the right one.
(Definitions: MAGI-1/inference/model/vae/vae_module.py, PatchEmbed L371 and ViTDecoder L656.)
"""

from pathlib import Path

import torch
import torch.nn as nn
from inference.model.vae import AutoModel, DiagonalGaussianDistribution
from inference.model.vae.vae_model import ViTVAE
from inference.model.vae.vae_module import ViTDecoder, ViTEncoder
from safetensors import safe_open
from safetensors.torch import load_file
from torch.utils.checkpoint import checkpoint

from magi_finetuning.settings import CKPT
from windinet.vae_adapter import InAdapter, OutAdapter


NUM_RGB = 3


VAE_CLIP_FRAMES = 16  # the VAE's native clip length (its position embedding covers 16 frames)


ADAPTER_HIDDEN = 32       # adapter width, activation and init as windinet's LTX colour-adapter run


ADAPTER_ACTIVATION = "gelu"


# ---------------------------------------------------------------- grown: both channel layers widened to 4 channels


def load_vae_4ch(init: str = "zeros") -> ViTVAE:
    """MAGI-1 VAE in fp32, both channel layers grown from 3 to 4.

    init says what the new (4th = pressure) channel starts as:
        "zeros": ignored by the encoder / predicted as 0 by the decoder; learned from scratch.
                 Keeps the pretrained RGB behaviour exactly at step 0 (your LTX default).
        "mean":  the average of the 3 RGB filters.
        "random": random filters (neither silent nor a copy): encoder input at the RGB filters' spread,
                 decoder output at 0.1x (a full-size random output filter would emit strong noise).
    Channels 0-2 (density, momentum_x, momentum_y) always start as copies of R, G, B.
    """
    vae: ViTVAE = AutoModel.from_pretrained(str(CKPT / "vae"))
    vae.encoder.patch_embed.proj = grow_input_channels(vae.encoder.patch_embed.proj, init)
    vae.decoder.last_layer = grow_output_channels(vae.decoder.last_layer, init)
    return vae.float()


def grow_input_channels(old: nn.Conv3d, init: str, n_channels: int = 4) -> nn.Conv3d:
    """Encoder's first layer: 3 input channels -> n_channels."""
    new = nn.Conv3d(n_channels, old.out_channels, old.kernel_size, old.stride, old.padding)
    with torch.no_grad():
        new.weight.zero_()
        new.weight[:, :NUM_RGB] = old.weight  # density, momentum_x, momentum_y take the R, G, B filters
        if init == "mean":
            new.weight[:, NUM_RGB:] = old.weight.mean(dim=1, keepdim=True)
        if init == "random":
            new.weight[:, NUM_RGB:].normal_(0.0, old.weight.std().item())
        new.bias.copy_(old.bias)  # bias is per output channel, unaffected
    return new


def grow_output_channels(old: nn.Conv3d, init: str, n_channels: int = 4) -> nn.Conv3d:
    """Decoder's last layer: 3 output channels -> n_channels."""
    new = nn.Conv3d(old.in_channels, n_channels, old.kernel_size, old.stride, old.padding)
    with torch.no_grad():
        new.weight.zero_()
        new.bias.zero_()
        new.weight[:NUM_RGB] = old.weight
        new.bias[:NUM_RGB] = old.bias
        if init == "mean":
            new.weight[NUM_RGB:] = old.weight.mean(dim=0, keepdim=True)
            new.bias[NUM_RGB:] = old.bias.mean()
        if init == "random":
            new.weight[NUM_RGB:].normal_(0.0, 0.1 * old.weight.std().item())  # small: no strong noise at step 0
            new.bias[NUM_RGB:] = old.bias.mean()
    return new


# ---------------------------------------------------------------- adapter: windinet's 1x1 adapters around the stock RGB VAE


class AdaptedEncoder(nn.Module):
    """4 fields -> InAdapter -> 3 channels -> MAGI's RGB encoder."""

    def __init__(self, encoder: ViTEncoder, n_channels: int = 4):
        super().__init__()
        self.in_adapter = InAdapter(n_channels, ADAPTER_HIDDEN, ADAPTER_ACTIVATION, identity_init=True)
        self.rgb_encoder = encoder

    def forward(self, x):
        return self.rgb_encoder(self.in_adapter(x))


class AdaptedDecoder(nn.Module):
    """MAGI's RGB decoder -> 3 channels -> OutAdapter -> 4 fields, clamped to [-1, 1] (as windinet)."""

    def __init__(self, decoder: ViTDecoder, n_channels: int = 4):
        super().__init__()
        self.rgb_decoder = decoder
        self.out_adapter = OutAdapter(n_channels, ADAPTER_HIDDEN, ADAPTER_ACTIVATION, identity_init=True)

    def forward(self, z):
        return self.out_adapter(self.rgb_decoder(z)).clamp(-1.0, 1.0)


class WideDecoder(nn.Module):
    """MAGI's ViT decoder with more values per output pixel.

    MAGI's decoder reshapes each 1024-dim token directly into 4 values for each of its 4x8x8 pixels
    (1024 / 256 = 4; vae_module.py ViTDecoder.forward) and one 3x3x3 conv (last_layer) turns those 4 values into
    the output fields. For RGB that is enough; for 4 independent physical fields it is a bottleneck: density and
    pressure compete for the same 4 numbers (official/B runs: pressure ~ density; AB: pressure learned, density
    lost). Here a linear head adds `extra` values per pixel from the same tokens, and last_layer reads 4 + extra.
    Optionally (refine > 0) a small residual conv stack on those 4 + extra values helps further: it has more than
    one pixel of reach to smooth the seams between the 8x8 patches, which are generated independently.

    At step 0 the output is exactly the grown VAE's: the new last_layer inputs and the refine branch's last conv
    start at zero; the head and the inner refine convs start small and random so gradients reach them right away.
    """

    def __init__(self, decoder: ViTDecoder, extra: int, refine: int = 0):
        super().__init__()
        self.vit = decoder
        self.extra = extra
        self.voxels = decoder.patch_length * decoder.patch_size * decoder.patch_size    # 4 x 8 x 8 = 256
        self.extra_proj = None
        if extra:
            self.extra_proj = nn.Linear(decoder.norm.normalized_shape[0], extra * self.voxels)
            nn.init.trunc_normal_(self.extra_proj.weight, std=0.02)                      # MAGI's own Linear init
            nn.init.zeros_(self.extra_proj.bias)
        old = decoder.last_layer
        self.last_layer = nn.Conv3d(old.in_channels + extra, old.out_channels, old.kernel_size, old.stride, old.padding)
        with torch.no_grad():
            self.last_layer.weight.zero_()
            self.last_layer.weight[:, :old.in_channels] = old.weight
            self.last_layer.bias.copy_(old.bias)
        decoder.last_layer = nn.Identity()  # the ViT now returns its 4 values per pixel
        self.refine = None
        if refine:
            width = old.in_channels + extra
            self.refine = nn.Sequential(
                nn.Conv3d(width, refine, 3, padding=1), nn.GELU(),
                nn.Conv3d(refine, refine, 3, padding=1), nn.GELU(),
                nn.Conv3d(refine, old.out_channels, 3, padding=1),
            )
            nn.init.zeros_(self.refine[-1].weight)
            nn.init.zeros_(self.refine[-1].bias)

    def forward(self, z):
        if self.extra_proj is None:  # refine only: no extra values per pixel
            features = self.vit(z)
            out = self.last_layer(features)
            return out if self.refine is None else out + self.refine(features)
        captured = {}
        hook = self.vit.norm.register_forward_hook(lambda module, inputs, output: captured.update(tokens=output))
        try:
            base = self.vit(z)  # (B, 4, T, H, W)
        finally:
            hook.remove()
        tokens = captured["tokens"][:, 1:] if self.vit.with_cls_token else captured["tokens"]
        B, _, T, H, W = z.shape
        p, q = self.vit.patch_length, self.vit.patch_size
        extra = self.extra_proj(tokens).reshape(B, T, H, W, p, q, q, self.extra)
        extra = extra.permute(0, 7, 1, 4, 2, 5, 3, 6).reshape(B, self.extra, T * p, H * q, W * q)  # as MAGI's rearrange
        features = torch.cat([base, extra], dim=1)  # 4 + extra values per pixel
        out = self.last_layer(features)
        return out if self.refine is None else out + self.refine(features)


def widen_decoder(vae: ViTVAE, extra: int, refine: int = 0) -> ViTVAE:
    """A grown 4-channel VAE whose decoder rebuilds each pixel from 4 + extra values, optionally with a residual
    refine conv stack of width `refine` (WideDecoder)."""
    vae.decoder = WideDecoder(vae.decoder, extra, refine)
    return vae


def load_vae_with_adapters() -> ViTVAE:
    """Stock MAGI-1 VAE in fp32 with windinet's identity-initialized adapters around it.

    At step 0 it behaves exactly like the RGB VAE on density, momentum_x, momentum_y (the adapters pass
    them through), and pressure is ignored by the encoder and predicted as 0 by the decoder, like
    load_vae_4ch("zeros"). vae.encoder / vae.decoder keep their call signatures, so the rest of the
    code (reconstruct, data.encode, evaluate.decode) works unchanged.
    """
    vae: ViTVAE = AutoModel.from_pretrained(str(CKPT / "vae"))
    vae.encoder = AdaptedEncoder(vae.encoder)
    vae.decoder = AdaptedDecoder(vae.decoder)
    return vae.float()


# ---------------------------------------------------------------- loading a trained VAE (train_vae.py checkpoints)


def vae_mode_of(vae_checkpoint: str) -> str:
    """"adapter", "wide" or "grown", from the checkpoint's metadata. Checkpoints without it (all runs before
    the adapter mode existed) are grown."""
    with safe_open(vae_checkpoint, framework="pt") as f:
        return (f.metadata() or {}).get("vae_mode", "grown")


def build_vae(mode: str, init: str = "zeros", decoder_extra_channels: int = 0, decoder_refine_channels: int = 0) -> ViTVAE:
    """The fp32 VAE architecture for a mode: "grown" (load_vae_4ch), "wide" (grown + WideDecoder) or "adapter"."""
    if mode == "adapter":
        return load_vae_with_adapters()
    vae = load_vae_4ch(init)
    if mode == "wide":
        return widen_decoder(vae, decoder_extra_channels, decoder_refine_channels).float()
    return vae


def load_trained_vae(vae_checkpoint: str) -> ViTVAE:
    """fp32 VAE from a train_vae.py checkpoint: the released weights, then the checkpoint's trained
    weights on top (an "io" checkpoint holds only the two channel layers; adapter/full ones more)."""
    with safe_open(vae_checkpoint, framework="pt") as f:
        metadata = f.metadata() or {}
    vae = build_vae(metadata.get("vae_mode", "grown"), "zeros", int(metadata.get("decoder_extra_channels", 0)),
                    int(metadata.get("decoder_refine_channels", 0)))
    _, unexpected = vae.load_state_dict(load_file(vae_checkpoint), strict=False)
    if unexpected:
        raise ValueError(f"{vae_checkpoint} has weights this VAE doesn't: {unexpected[:5]}")
    return vae


def check_vae_finished(vae_checkpoint: str) -> None:
    """A VAE from a train_vae.py run (it has timing.json) must have finished (finished.json); otherwise a
    DiT would silently train on an intermediate VAE. Ad-hoc checkpoints without timing.json are allowed."""
    run_dir = Path(vae_checkpoint).parent
    if (run_dir / "timing.json").is_file() and not (run_dir / "finished.json").is_file():
        raise RuntimeError(f"VAE run in {run_dir} has not finished (no finished.json); "
                           "refusing to use its intermediate checkpoint")


def load_vae_encoder(vae_checkpoint: str | None = None):
    """Frozen 4-channel encoder for DiT training (bf16). With vae_checkpoint (from train_vae.py): the adapted
    VAE from stage 1. Without: the released encoder, pressure channel = mean of the RGB filters."""
    if vae_checkpoint:
        check_vae_finished(vae_checkpoint)
        vae = load_trained_vae(vae_checkpoint)
    else:
        vae = load_vae_4ch("mean")
    return vae.encoder.cuda().bfloat16().eval().requires_grad_(False)


def load_vae(vae_checkpoint: str | None):
    """Frozen 4-channel VAE (encoder + decoder), bf16, as the DiT saw it during training.

    With a checkpoint from train_vae.py: its trained weights on top of the released ones (grown or adapter).
    Without: the same un-adapted encoder the DiT path uses (RGB filters + their mean for pressure);
    the decoder's pressure output is then only a mean-of-RGB guess.
    """
    if vae_checkpoint:
        check_vae_finished(vae_checkpoint)
        vae = load_trained_vae(vae_checkpoint)
    else:
        vae = load_vae_4ch("mean")
    return vae.cuda().bfloat16().eval().requires_grad_(False)


# ---------------------------------------------------------------- training and running


def enable_vae_gradient_checkpointing(vae: ViTVAE) -> None:
    """Recompute each transformer block of the encoder and decoder in backward instead of storing it.
    Finds them inside the adapter wrappers too."""
    for module in vae.modules():
        if isinstance(module, (ViTEncoder, ViTDecoder)):
            for block in module.blocks:
                block.forward = lambda *a, _f=block.forward, **k: checkpoint(_f, *a, use_reentrant=False, **k)


def reconstruct(vae: ViTVAE, video: torch.Tensor):
    """video (B, 4, T, H, W) in [-1, 1], T a multiple of 16 -> (reconstruction, latent (B, 16, T/4, h, w)).

    Encoded and decoded in independent 16-frame tiles, exactly as the DiT path encodes (data.encode)
    and evaluation decodes. The latent is the posterior mean (no sampling), as MAGI-1 encodes.
    """
    if video.shape[2] % VAE_CLIP_FRAMES:
        raise ValueError(f"{video.shape[2]} frames is not a multiple of {VAE_CLIP_FRAMES}")
    latents, reconstructions = [], []
    for start in range(0, video.shape[2], VAE_CLIP_FRAMES):
        latent = DiagonalGaussianDistribution(vae.encoder(video[:, :, start:start + VAE_CLIP_FRAMES])).mode()
        latents.append(latent)
        reconstructions.append(vae.decoder(latent))
    return torch.cat(reconstructions, dim=2), torch.cat(latents, dim=2)
