"""Checks for data.Normalization, utils/resume.py, the adapter VAE, the windowed sampler and clean-chunk
training (one GPU, a few minutes).

    sbatch jobs/jupiter/magi1/quick_test.sbatch -m magi_finetuning.tests.test_changes [--old-vae <grown ckpt>]

Each check prints PASS/FAIL; the last line says whether all passed.
"""

import argparse
import tempfile
from pathlib import Path

import torch
import yaml

import magi_finetuning.compat  # noqa: F401  (stand-ins before MAGI imports)
from magi_finetuning.data import DEFAULT, Normalization
from magi_finetuning.settings import NORMALIZATION_CLIP
from windinet.training.shockwave_data import normalize_fields

RESULTS = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)


def physical_fields(shape=(1, 4, 8, 32, 32)) -> torch.Tensor:
    """Plausible positive density / pressure, signed momenta."""
    x = torch.randn(shape) * 0.5
    x[:, 0] = x[:, 0].abs() + 0.05
    x[:, 3] = x[:, 3].abs() + 0.05
    return x


def test_normalization() -> None:
    x = physical_fields()
    mean, std = DEFAULT.mean_std
    old = normalize_fields(x, mean, std, NORMALIZATION_CLIP)
    check("default normalization == old normalize_fields (bit for bit)", torch.equal(DEFAULT.apply(x), old))
    error = (DEFAULT.invert(DEFAULT.apply(x)) - x).abs().max().item()
    check("default invert(apply(x)) == x", error < 1e-5, f"max err {error:.1e}")

    # log mode, with a temporary yaml that has log_pressure (the real one comes from channel_stats.py)
    stats = yaml.safe_load(open(DEFAULT.stats_file))
    stats["data_normalization_stats"]["log_pressure"] = {"mean": -0.3, "std": 0.7}
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(stats, f)
    logged = Normalization(f.name, ("density", "pressure"))
    y = logged.apply(x)
    expected_density = ((x[:, 0].log() - stats["data_normalization_stats"]["log_density"]["mean"])
                        / (stats["data_normalization_stats"]["log_density"]["std"] * NORMALIZATION_CLIP)).clamp(-1, 1)
    check("log mode normalizes log(density) with log_density stats",
          torch.allclose(y[:, 0], expected_density, atol=1e-6))
    check("log mode leaves momenta as before", torch.equal(y[:, 1:3], DEFAULT.apply(x)[:, 1:3]))
    error = (logged.invert(y) - x).abs().max().item()
    check("log mode invert(apply(x)) == x", error < 1e-4, f"max err {error:.1e}")
    check("metadata round trip", Normalization.from_metadata(logged.to_metadata()) == logged)
    check("no metadata -> DEFAULT", Normalization.from_metadata({}) == DEFAULT)
    try:
        Normalization(DEFAULT.stats_file, ("pressure",)).mean_std
        check("missing log_pressure in old yaml raises", False)
    except KeyError:
        check("missing log_pressure in old yaml raises", True)


def test_resumable_order() -> None:
    from accelerate import Accelerator
    from torch.utils.data import DataLoader, TensorDataset

    from magi_finetuning.utils.resume import EpochShuffle, batches_from

    accelerator = Accelerator()
    dataset = TensorDataset(torch.arange(10))
    sampler = EpochShuffle(len(dataset), seed=7)
    loader = accelerator.prepare(DataLoader(dataset, batch_size=1, sampler=sampler))
    stream = batches_from(loader, accelerator, 0, epoch_aware=[sampler])  # (start_epoch, not set_epoch)
    uninterrupted = [int(next(stream)[0]) for _ in range(25)]
    epochs_differ = uninterrupted[:10] != uninterrupted[10:20]
    check("EpochShuffle: every epoch a permutation, epochs differ",
          sorted(uninterrupted[:10]) == list(range(10)) and epochs_differ, str(uninterrupted[:12]))

    resumed_sampler = EpochShuffle(len(dataset), seed=7)
    resumed_loader = accelerator.prepare(DataLoader(dataset, batch_size=1, sampler=resumed_sampler))
    resumed = batches_from(resumed_loader, accelerator, 13, epoch_aware=[resumed_sampler])
    continued = [int(next(resumed)[0]) for _ in range(12)]
    check("batches_from(13) continues exactly where an uninterrupted run is", continued == uninterrupted[13:25],
          f"{continued[:6]} vs {uninterrupted[13:19]}")


class _FakeAccelerator:
    is_main_process = True

    def unwrap_model(self, model):
        return model

    def print(self, *args):
        print(*args)


class _Holder(torch.nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.magi_vae = vae


@torch.no_grad()
@torch.autocast("cuda", dtype=torch.bfloat16)  # MAGI's VAE attention runs in bf16 only, as in training/evaluation
def test_adapter_vae(old_vae: str | None) -> None:
    from inference.model.vae import AutoModel

    from magi_finetuning.settings import CKPT
    from magi_finetuning.train.train_vae import save_weights
    from magi_finetuning.vae import load_trained_vae, load_vae_with_adapters, vae_mode_of

    stock = AutoModel.from_pretrained(str(CKPT / "vae")).float().cuda().eval()
    adapted = load_vae_with_adapters().cuda().eval()
    x = (torch.rand(1, 4, 16, 64, 64, device="cuda") * 2 - 1)
    z_stock, z_adapted = stock.encoder(x[:, :3]), adapted.encoder(x)
    check("adapter VAE at init: encoder == stock encoder on the 3 fields (pressure ignored)",
          torch.equal(z_stock, z_adapted), f"max diff {(z_stock - z_adapted).abs().max().item():.1e}")
    latent = z_stock[:, :16]
    out_stock, out_adapted = stock.decoder(latent).clamp(-1, 1), adapted.decoder(latent)
    check("adapter VAE at init: decoder == stock decoder, pressure 0",
          torch.equal(out_stock, out_adapted[:, :3]) and out_adapted[:, 3].abs().max() == 0)
    trainable = sum(p.numel() for p in adapted.encoder.in_adapter.parameters())
    check("input adapter is 4->32->3 (1x1x1 convs)", trainable == 4 * 32 + 32 + 32 * 3 + 3, f"{trainable} params")

    # save as train_vae does (trained weights + metadata), reload, same outputs
    for p in adapted.encoder.in_adapter.parameters():
        p.add_(0.01 * torch.randn_like(p))
    logged = Normalization(DEFAULT.stats_file, ("density",))
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "vae_best.safetensors"
        save_weights(_FakeAccelerator(), _Holder(adapted), path, (Path(folder), "adapter", logged, "adapter", (0, 0)))
        reloaded = load_trained_vae(str(path)).cuda().eval()
        check("adapter checkpoint: vae_mode metadata", vae_mode_of(str(path)) == "adapter")
        check("adapter checkpoint: normalization metadata", Normalization.of_vae_checkpoint(str(path)) == logged)
        # trained weights come back up to bf16 storage precision (relative 2^-8)
        mine, theirs = dict(adapted.named_parameters()), dict(reloaded.named_parameters())
        trained = [n for n in mine if n.startswith(("encoder.in_adapter.", "decoder."))]
        worst = max(((mine[n] - theirs[n]).abs().max() / mine[n].abs().max().clamp_min(1e-12)).item() for n in trained)
        check("adapter checkpoint reloads every trained weight (bf16 storage)", worst < 2**-7, f"worst rel err {worst:.1e}")

    if old_vae:
        check("older grown checkpoint: mode 'grown'", vae_mode_of(old_vae) == "grown")
        check("older grown checkpoint: DEFAULT normalization", Normalization.of_vae_checkpoint(old_vae) == DEFAULT)
        grown = load_trained_vae(old_vae).cuda().eval()
        check("older grown checkpoint loads", grown.encoder.patch_embed.proj.in_channels == 4)


@torch.no_grad()
@torch.autocast("cuda", dtype=torch.bfloat16)
def test_wide_decoder() -> None:
    from magi_finetuning.train.train_vae import save_weights
    from magi_finetuning.vae import build_vae, load_trained_vae, vae_mode_of

    torch.manual_seed(0)
    grown = build_vae("grown").cuda().eval()
    torch.manual_seed(0)
    wide = build_vae("wide", decoder_extra_channels=16, decoder_refine_channels=64).cuda().eval()
    z = torch.randn(1, 16, 4, 8, 8, device="cuda")
    out_grown, out_wide = grown.decoder(z), wide.decoder(z)
    check("wide decoder + refine at init == grown decoder (new parts start at zero)", torch.equal(out_grown, out_wide),
          f"max diff {(out_grown - out_wide).abs().max().item():.1e}, shape {tuple(out_wide.shape)}")
    check("wide decoder: last layer reads 4 + 16 channels", wide.decoder.last_layer.in_channels == 20)

    for p in [wide.decoder.extra_proj.weight, wide.decoder.last_layer.weight, wide.decoder.refine[-1].weight]:
        p.add_(0.01 * torch.randn_like(p))
    for p in wide.parameters():  # full checkpoints store bf16: round first, so a correct reload is exact
        p.copy_(p.bfloat16().float())
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "vae_best.safetensors"
        save_weights(_FakeAccelerator(), _Holder(wide), path, (Path(folder), "full", DEFAULT, "wide", (16, 64)))
        reloaded = load_trained_vae(str(path)).cuda().eval()
        check("wide checkpoint: vae_mode metadata", vae_mode_of(str(path)) == "wide")
        mine, theirs = dict(wide.named_parameters()), dict(reloaded.named_parameters())
        same = mine.keys() == theirs.keys() and all(torch.equal(mine[n], theirs[n]) for n in mine)
        check("wide checkpoint reloads every weight exactly", same, f"{len(mine)} parameters")


class _VelocityStandIn:
    """Deterministic stand-in for dit.Model: each chunk's velocity depends on its own t and on the mean of
    every chunk it may attend to (0..k), so a wrong context or noise level changes the result. Records the
    chunk times of every call."""

    def __init__(self):
        self.calls = []

    def __call__(self, x, t, gamma, n_chunks, tokens_per_chunk):
        from magi_finetuning.settings import CHUNK_WIDTH
        self.calls.append(t[0].clone())
        chunks = x.unflatten(2, (n_chunks, CHUNK_WIDTH))                       # (1, C, n, 6, h, w)
        means = chunks.mean(dim=(3, 4, 5), keepdim=True)
        context = means.cumsum(dim=2) / torch.arange(1, n_chunks + 1, device=x.device).view(1, 1, -1, 1, 1, 1)
        tf = t.view(1, 1, n_chunks, 1, 1, 1)
        return (torch.tanh(chunks) * (1 - tf) + 0.3 * context + 0.1 * gamma).flatten(2, 3)


def sequential_rollout_before_2026_10_28(model, image_latent, gamma, num_chunks, tokens_per_chunk, num_steps, shift,
                                         clean_t):
    """dit.rollout as it was until 2026-10-28 (one chunk at a time), kept to check window = 1 against it."""
    from magi_finetuning.dit import time_schedule
    from magi_finetuning.settings import CHUNK_WIDTH
    times = time_schedule(num_steps, shift, image_latent.device)
    chunks = []
    for k in range(num_chunks):
        x = torch.randn(image_latent.shape[:2] + (CHUNK_WIDTH,) + image_latent.shape[3:], device=image_latent.device)
        for i in range(num_steps):
            if k == 0:
                x[:, :, :1] = image_latent
            chunk_times = torch.full((1, k + 1), clean_t, device=x.device)
            chunk_times[:, -1] = times[i]
            velocity = model(torch.cat(chunks + [x], dim=2), chunk_times, gamma, k + 1, tokens_per_chunk)
            x = x + velocity[:, :, -CHUNK_WIDTH:].float() * (times[i + 1] - times[i])
        if k == 0:
            x[:, :, :1] = image_latent
        chunks.append(x)
    return torch.cat(chunks, dim=2)


def test_windowed_sampler() -> None:
    from magi_finetuning.dit import rollout, time_schedule
    from magi_finetuning.settings import CHUNK_WIDTH
    image = torch.randn(1, 16, 1, 4, 4, device="cuda")
    gamma = torch.tensor([[1.4]], device="cuda")
    setup = dict(num_chunks=5, tokens_per_chunk=0, num_steps=64, shift=3.0, clean_t=0.9999)

    torch.manual_seed(0)
    old = sequential_rollout_before_2026_10_28(_VelocityStandIn(), image, gamma, **setup)
    torch.manual_seed(0)
    new = rollout(_VelocityStandIn(), image, gamma, window=1, **setup)
    check("rollout window 1 == the old one-chunk-at-a-time rollout (bit for bit)", torch.equal(old, new),
          f"max diff {(old - new).abs().max().item():.1e}")

    model = _VelocityStandIn()
    torch.manual_seed(0)
    out = rollout(model, image, gamma, window=4, **setup)
    times = time_schedule(64, 3.0, "cuda")
    check("window 4: 8 stages x 16 steps = 128 model calls", len(model.calls) == 128, f"{len(model.calls)} calls")
    per_chunk = [[] for _ in range(5)]   # the t each chunk was denoised at, call by call
    staircase, finished_first = True, True
    for call in model.calls:
        active = [k for k in range(len(call)) if call[k] != 0.9999]
        for k in active:
            per_chunk[k].append(call[k].item())
        staircase &= all(bool(call[a] > call[b]) for a, b in zip(active, active[1:]))  # earlier chunk cleaner
        finished_first &= active == list(range(active[0], len(call)))  # clean chunks only before the active ones
    every_step_once = all(torch.allclose(torch.tensor(steps), times[:-1].cpu()) for steps in per_chunk)
    check("window 4: every chunk runs all 64 steps of the schedule once, in order", every_step_once,
          f"steps per chunk {[len(steps) for steps in per_chunk]}")
    check("window 4: active chunks form a staircase (earlier = cleaner), finished ones are clean context",
          staircase and finished_first)
    check("window 4: at most 4 chunks denoised at once",
          max(int((call != 0.9999).sum()) for call in model.calls) == 4)
    check("window 4: frame 0 stays the given image latent", torch.equal(out[:, :, :1], image))
    first_call_of_chunk1 = next(c for c in model.calls if len(c) == 2)
    check("window 4: chunk 1 starts when chunk 0 is 16 of 64 steps in (t 0.10)",
          abs(first_call_of_chunk1[0].item() - times[16].item()) < 1e-6,
          f"chunk 0 at t {first_call_of_chunk1[0].item():.3f}")
    check("window 4 output shape", out.shape == (1, 16, 5 * CHUNK_WIDTH, 4, 4))


def test_clean_chunk_training() -> None:
    from magi_finetuning.dit import CLEAN_CHUNK_NOISE, sample_chunk_t, sample_num_clean_chunks, velocity_loss
    from magi_finetuning.settings import CHUNK_WIDTH

    torch.manual_seed(0)
    counts = torch.bincount(torch.tensor([sample_num_clean_chunks(5) for _ in range(40000)]), minlength=5) / 40000
    expected = torch.tensor([0.5, 0.125, 0.125, 0.125, 0.125])
    check("clean chunks drawn as often as window 4 meets them (0: 1/2, 1..4: 1/8 each)",
          bool((counts - expected).abs().max() < 0.01), f"observed {[round(c, 3) for c in counts.tolist()]}")

    t = sample_chunk_t(5, 3.0, num_clean=2)[0]
    clean_ok = bool(((t[:2] >= 1 - CLEAN_CHUNK_NOISE) & (t[:2] <= 1)).all())
    check("sample_chunk_t: clean chunks at t in [0.95, 1], noisy ones sorted (earlier cleaner)",
          clean_ok and bool((t[2:-1] >= t[3:]).all()) and t.shape == (5,), f"t = {[round(x, 3) for x in t.tolist()]}")
    torch.manual_seed(1)
    u = torch.rand(1, 5, device="cuda")
    before = (u / (3.0 - 2.0 * u)).sort(dim=1, descending=True).values  # sample_chunk_t until 2026-10-28
    torch.manual_seed(1)
    check("sample_chunk_t without clean chunks == the old draw (bit for bit)", torch.equal(before, sample_chunk_t(5, 3.0)))

    num_body, latents = 25, 30
    x1, noise = torch.zeros(1, 16, latents, 2, 2), torch.zeros(1, 16, latents, 2, 2)
    v = torch.arange(latents, dtype=torch.float).view(1, 1, latents, 1, 1).expand(1, 16, latents, 2, 2)  # error = index

    def mean_sq(first):  # the loss if exactly latents first..25 are scored
        return (torch.arange(first, num_body + 1, dtype=torch.float) ** 2).mean().item()

    check("velocity_loss: no clean chunks -> latents 1..25",
          abs(velocity_loss(v, x1, noise, num_body).item() - mean_sq(1)) < 1e-3)
    check("velocity_loss: 2 clean chunks -> latents 12..25 only",
          abs(velocity_loss(v, x1, noise, num_body, 2).item() - mean_sq(2 * CHUNK_WIDTH)) < 1e-3)
    check("velocity_loss: 4 clean chunks -> latents 24..25 (frames 93..100)",
          abs(velocity_loss(v, x1, noise, num_body, 4).item() - mean_sq(24)) < 1e-3)


def test_context_noise_and_schedule() -> None:
    from types import SimpleNamespace

    from magi_finetuning.settings import CHUNK_WIDTH
    from magi_finetuning.train.train_full import make_noisy_input, make_scheduler

    num_body, latents, clean_t = 25, 30, 0.9999
    torch.manual_seed(0)
    x1 = (torch.randn(1, 16, latents, 4, 4, device="cuda") * torch.linspace(0.5, 2.0, 16, device="cuda").view(1, 16, 1, 1, 1))
    off = SimpleNamespace(clean_chunks=True, shift=3.0, context_noise=0.0)
    on = SimpleNamespace(clean_chunks=True, shift=3.0, context_noise=0.25)
    seen = set()
    for _ in range(400):
        noisy, t, noise, num_clean = make_noisy_input(x1, 5, num_body, on, clean_t)
        seen.add(num_clean)
        context = slice(1, num_clean * CHUNK_WIDTH)
        rest = slice(max(1, num_clean * CHUNK_WIDTH), latents)
        tf = t.repeat_interleave(CHUNK_WIDTH, dim=1)[:, None, :, None, None]
        expected_rest = (tf * x1 + (1 - tf) * noise)[:, :, rest]
        if not (torch.equal(noisy[:, :, :1], x1[:, :, :1]) and torch.allclose(noisy[:, :, rest], expected_rest)
                and bool((t[0, :num_clean] == clean_t).all())):
            check("context noise: latent 0 clean, noisy chunks from the exact clean latents, context at clean_t", False)
            return
        if num_clean:
            ratio = ((noisy - x1)[:, :, context].std(dim=(0, 2, 3, 4)) / x1[:, :, 1:1 + num_body].std(dim=(0, 2, 3, 4)))
            if not (ratio.max() <= 0.25 * 1.3 and (ratio.max() - ratio.min()) / ratio.mean().clamp_min(1e-6) < 0.6):
                check("context noise: sigma * std(z) per channel, sigma <= 0.25", False, f"ratios {ratio.tolist()}")
                return
    check("context noise: latent 0 clean, noisy chunks from the exact clean latents, context at clean_t, "
          "per-channel sigma * std(z) with sigma <= 0.25", seen == {0, 1, 2, 3, 4}, f"clean counts seen {sorted(seen)}")
    torch.manual_seed(3)
    a = make_noisy_input(x1, 5, num_body, off, clean_t)
    torch.manual_seed(3)
    from magi_finetuning.dit import add_noise, keep_image_clean, sample_chunk_t, sample_num_clean_chunks
    num_clean = sample_num_clean_chunks(5)
    t = sample_chunk_t(5, 3.0, num_clean)
    noise = torch.randn_like(x1)
    check("context noise 0: same draw as before (bit for bit)",
          torch.equal(a[0], keep_image_clean(add_noise(x1, t, noise), x1)) and torch.equal(a[1], t))

    def lrs(warmup, steps=2000, lr=2e-6, lr_min=1e-7):
        opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=lr)
        sched = make_scheduler(opt, SimpleNamespace(steps=steps, warmup=warmup, lr=lr, lr_min=lr_min))
        out = []
        for _ in range(steps):
            out.append(sched.get_last_lr()[0])
            opt.step()
            sched.step()
        return out, type(sched).__name__

    cos, kind = lrs(0, steps=8000, lr=1e-5, lr_min=1e-8)
    check("schedule, warmup 0: CosineAnnealingLR from lr (the earlier runs' schedule)",
          kind == "CosineAnnealingLR" and cos[0] == 1e-5 and abs(cos[4000] - (1e-5 + 1e-8) / 2) < 1e-9)
    w, _ = lrs(100)
    check("schedule, warmup 100: lr/100 at step 0, 2e-6 at step 100, cosine to ~1e-7 at step 2000",
          abs(w[0] - 2e-8) < 1e-12 and abs(w[50] - 1.01e-6) < 1e-10 and abs(w[100] - 2e-6) < 1e-12
          and abs(w[1050] - 1.05e-6) < 1e-9 and abs(w[-1] - 1e-7) < 1e-9 and all(a >= b for a, b in zip(w[100:], w[101:])),
          f"lr at 0 / 50 / 100 / 1050 / 1999: {w[0]:.3g} {w[50]:.3g} {w[100]:.3g} {w[1050]:.3g} {w[-1]:.3g}")


def test_latent_prior() -> None:
    """--latent-prior: noise with the latents' per-channel mean / std in training and at the start of sampling,
    the same random draws as without it, loss weights 1 / std^2 with mean 1."""
    from types import SimpleNamespace

    from magi_finetuning.dit import LatentPrior, rollout, velocity_loss
    from magi_finetuning.settings import SCRATCH
    from magi_finetuning.train.train_full import make_noisy_input

    prior = LatentPrior(str(SCRATCH / "magi1_latents_F3" / "stats.json"), 0.18215)
    check("latent prior: mean(std^2) = 1 and mean loss weight = 1",
          abs(prior.std.pow(2).mean().item() - 1) < 1e-5 and abs(prior.loss_weight.mean().item() - 1) < 1e-5,
          f"std {prior.std.min().item():.2f}..{prior.std.max().item():.2f}")
    args = SimpleNamespace(clean_chunks=False, shift=3.0, context_noise=0.0)
    x1 = torch.randn(1, 16, 30, 32, 32, device="cuda")
    torch.manual_seed(5)
    plain_noisy, plain_t, plain_noise, _ = make_noisy_input(x1, 5, 25, args, 0.9999)
    torch.manual_seed(5)
    noisy, t, noise, _ = make_noisy_input(x1, 5, 25, args, 0.9999, prior)
    tf = t.repeat_interleave(6, dim=1)[:, None, :, None, None]
    check("latent prior: same t and random draws as without it, noise = mean + std * eps, x_t = t x + (1 - t) noise",
          torch.equal(t, plain_t) and torch.allclose(noise, prior.noise(plain_noise))
          and torch.allclose(noisy[:, :, 1:], (tf * x1 + (1 - tf) * noise)[:, :, 1:]) and torch.equal(noisy[:, :, :1], x1[:, :, :1]))
    v = torch.randn_like(x1)
    weighted = velocity_loss(v, x1, noise, 25, 0, prior.loss_weight).item()
    manual = (((v - (x1 - noise))[:, :, 1:26] ** 2) / prior.std ** 2).mean().item() / prior.std.pow(-2).mean().item()
    check("latent prior: loss = squared error / std^2 (weights scaled to mean 1)", abs(weighted - manual) < 1e-4 * manual)
    still = lambda x, times, gamma, chunks, tokens: torch.zeros_like(x)  # zero velocity: the output is the starting noise
    image = torch.full((1, 16, 1, 32, 32), 7.0, device="cuda")
    start = rollout(still, image, None, 5, 1536, num_steps=4, prior=prior)[:, :, 1:]
    check("latent prior: the sampler starts every chunk from the prior's noise",
          (start.mean(dim=(0, 2, 3, 4)) - prior.mean.flatten()).abs().max().item() < 0.05
          and (start.std(dim=(0, 2, 3, 4)) - prior.std.flatten()).abs().max().item() < 0.05)


def test_standard_video(out_dir) -> None:
    """Acceptance checks of docs/video_spec.md on windinet/video.py with the shared statistics."""
    import json
    import subprocess
    from pathlib import Path

    import imageio_ffmpeg
    import numpy as np

    from magi_finetuning.settings import OUTPUTS
    from windinet.video import RolloutFigure, scale, write_video

    stats = json.loads((OUTPUTS / "rollout_video_stats.json").read_text())
    names = list(stats)
    channels = [stats[c] for c in names]
    check("video stats: signed exactly for momentum", [c.get("signed", False) for c in channels] == [False, True, True, False])
    density, momentum = scale(stats["density"]), scale(stats["momentum_x"])
    check("video scale: density mean +- 2 std, not below min; viridis",
          abs(density.lo - max(stats["density"]["mean"] - 2 * stats["density"]["std"], stats["density"]["min"])) < 1e-9
          and abs(density.hi - (stats["density"]["mean"] + 2 * stats["density"]["std"])) < 1e-9 and density.cmap == "viridis",
          f"lo {density.lo:.3f} hi {density.hi:.3f}")
    check("video scale: momentum symmetric around 0; RdBu_r",
          momentum.lo == -momentum.hi and momentum.centre == 0.0 and momentum.cmap == "RdBu_r", f"+-{momentum.hi:.3f}")

    rng = np.random.default_rng(0)
    t, h = 5, 32
    gt = np.stack([stats[c]["mean"] + stats[c]["std"] * rng.standard_normal((t, h, h)) for c in names], axis=1).astype("float32")
    times = 0.015 * np.arange(t)
    figure = RolloutFigure(names, channels, (h, h), "test")
    same, numbers = figure.frame(0.0, gt[0], gt[0])
    panel = np.asarray(figure.images[0][2].get_array())
    check("video: prediction = truth gives a zero error panel, RMSE 0", bool((panel == 0).all()) and numbers[0]["rmse"] == 0)
    check("video frame: 960 x 1090 px for 4 channels", same.shape == (1090, 960, 3), f"{same.shape}")
    bad = gt[1].copy()
    bad[0, 0, 0] = np.nan
    bad[1, 0, 1] = 1e4
    _, numbers = figure.frame(0.015, gt[1], bad)
    check("video: non-finite counted and RMSE undefined, out-of-range counted",
          numbers[0]["rmse"] is None and numbers[0]["fraction_nonfinite"] == 1 / h ** 2
          and numbers[1]["fraction_error_beyond"] >= 1 / h ** 2 and numbers[1]["fraction_prediction_beyond"] >= 1 / h ** 2)
    import matplotlib.pyplot as plt
    plt.close(figure.fig)

    path = Path(out_dir) / "video_check"
    path.parent.mkdir(parents=True, exist_ok=True)
    pred = gt + 0.1 * np.array([stats[c]["std"] for c in names], dtype="float32")[None, :, None, None]
    frames = write_video(gt, pred, times, names, channels, path, title="test", save_json=True)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    info = subprocess.run([ffmpeg, "-i", str(path.with_suffix(".mp4"))], capture_output=True, text=True).stderr
    # ffmpeg -i prints only the first 255 characters of a tag: read the whole comment
    comment = subprocess.run([ffmpeg, "-loglevel", "error", "-i", str(path.with_suffix(".mp4")), "-f", "ffmetadata", "-"],
                             capture_output=True, text=True).stdout
    check("video file: 960x1090, slowdown 8 -> 8.33 fps, stats in the mp4 comment, json per frame",
          "960x1090" in info and "8.33 fps" in info and "momentum_x" in comment and "AsinhNorm" in comment and len(frames) == t
          and len(json.loads(path.with_suffix(".json").read_text())["frames"]) == t, info[-300:] if "8.33 fps" not in info else "")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--old-vae", default=None, help="a VAE checkpoint from before the metadata existed")
    p.add_argument("--video-out", default="/tmp/standard_video_check", help="folder for the short test video")
    args = p.parse_args()
    test_normalization()
    test_resumable_order()
    test_adapter_vae(args.old_vae)
    test_wide_decoder()
    test_windowed_sampler()
    test_clean_chunk_training()
    test_context_noise_and_schedule()
    test_latent_prior()
    test_standard_video(args.video_out)
    print(f"\n{sum(RESULTS)}/{len(RESULTS)} checks passed -> {'ALL PASS' if all(RESULTS) else 'FAILURES'}", flush=True)
