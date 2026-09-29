#!/usr/bin/env python
"""VAE-only reconstruction eval on the standalone 256x256 test set. Not a training run.

Chapter 6 compares every VAE fine-tuning arm on one common held-out set:
the Euler_MQ ``256x256_ds/test.h5`` file. That file is a sibling of train.h5
and was never part of any train/val split or tuning decision. The per-run
``val_vrmse`` in metrics.csv is not usable for this comparison: it was
measured on each run's own validation slice of train.h5, at each run's own
training resolution (the 128res arm validates on 128x128 data), and it also
picked the checkpoint.

Metrics are the same ones VaeTrainer._evaluate logs, computed with the same
functions (vrmse / vrmse_per_channel from windinet.training.vae_trainer):

  * ``vrmse``         -- pooled over all channels (the historical val_vrmse)
  * ``vrmse_chmean``  -- mean of the four per-channel VRMSEs (headline metric)
  * ``vrmse_<field>`` -- per-channel breakdown

They are accumulated per sim and then averaged, so the result does not depend
on --batch-size. Per-sim values are written to the JSON as well, which allows
paired comparisons and bootstrap CIs between arms without re-running.

Encode/decode reproduce VaeTrainer._encode/_decode exactly: latents_mean/std
rescale, scaling_factor, cfg.adapter.default_temb, and the config's
mixed-precision autocast. The VAE is rebuilt the same way VaeTrainer._load_vae
builds it: mode='inflate' grows conv_in/conv_out and then loads the
ltx-inflated-io-v1 checkpoint, while mode='adapter' wraps the frozen VAE in
in/out adapters loaded from the checkpoint. Input normalization comes from the
run's own config. For the 128res arm that means the 128x128 stats file it was
trained with, applied to 256x256 data. Those stats are within 0.2% of the
256x256 ones, so every arm sees effectively the same normalized targets.

Usage:
    python scripts/eval_vae_test.py configs/finetune_vae/finetune_vae_ch6_loss_rmse_h1_256res.yaml \\
        --checkpoint $SCRATCH/windinet/finetune_vae_outputs_sng_pvc/finetune_vae_ch6_loss_rmse_h1_256res/checkpoints/vae_shockwave_best.safetensors \\
        --test-h5 $SCRATCH/windinet/euler_mq_dataset/256x256_ds/test.h5 \\
        --output finetune_vae_outputs/sng_pvc/finetune_vae_ch6_loss_rmse_h1_256res/test_eval_256.json

Figure data: for --save-samples sims spanning the test set's gamma range
(min/median/max for 3, the same picks for every arm and for
scripts/eval_dit_vrmse.py --test_h5), the fields at --sample-frames are saved
as one float16 .npz per sim: gt_raw (physical units), gt_norm and recon_norm
(the normalized/clipped space the metrics use), plus the normalization stats.
They go to $SCRATCH/windinet/figure_data/vae_test/<run>/ by default, not
into the git-tracked run mirror (~5 MB per sim).

Single tile, plain python (no accelerate launch): forward passes only.
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import typer
import yaml
from rich.console import Console
from rich.table import Table
from torch.utils.data import DataLoader, Subset

from windinet.config import VaeTrainerConfig
from windinet.inference.model_loader import load_vae
from windinet.training.shockwave_data import ShockWaveDataset, build_shockwave_video, pick_gamma_spread_ids
from windinet.training.vae_trainer import vrmse, vrmse_per_channel
from windinet.utils import get_default_device
from windinet.vae_adapter import inflate_vae_io_channels, load_adapted_vae, load_inflated_vae_checkpoint

console = Console()
app = typer.Typer(pretty_exceptions_enable=False, no_args_is_help=True)

_AUTOCAST_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}


def _build_vae(cfg: VaeTrainerConfig, checkpoint: str | None, device: torch.device):
    """Rebuild the VAE exactly as VaeTrainer._load_vae does, with `checkpoint` as its weights."""
    vae = load_vae(cfg.model.model_source, dtype=torch.float32)
    adapter_cfg = cfg.adapter
    if adapter_cfg.enabled and adapter_cfg.mode == "adapter":
        if checkpoint is None:
            raise typer.BadParameter("mode='adapter' needs --checkpoint (there is no untrained adapter reference)")
        vae, _ = load_adapted_vae(
            vae,
            ckpt_path=checkpoint,
            device="cpu",
            dtype=torch.float32,
            channels=adapter_cfg.channels,
            k=adapter_cfg.hidden_channels,
            activation=adapter_cfg.activation,
            identity_init=adapter_cfg.identity_init,
            default_temb=adapter_cfg.default_temb,
        )
    elif adapter_cfg.enabled and adapter_cfg.mode == "inflate":
        copy_from_index = (
            adapter_cfg.channels.index(adapter_cfg.inflate_copy_channel)
            if adapter_cfg.inflate_init == "copy"
            else None
        )
        inflate_vae_io_channels(vae, n=len(adapter_cfg.channels), init=adapter_cfg.inflate_init, copy_from_index=copy_from_index)
        if checkpoint is not None:
            meta = load_inflated_vae_checkpoint(vae, ckpt_path=checkpoint, device="cpu", dtype=torch.float32)
            console.print(f"  loaded inflate checkpoint (epoch={meta.get('epoch', '?')}): {checkpoint}")
        else:
            console.print("  no checkpoint: freshly inflated pretrained VAE")
    else:
        raise typer.BadParameter(f"unsupported adapter config (enabled={adapter_cfg.enabled}, mode={adapter_cfg.mode!r})")
    for p in vae.parameters():
        p.requires_grad_(False)
    vae.eval()
    return vae.to(device)


def _encode(vae, video: torch.Tensor) -> torch.Tensor:
    """VaeTrainer._encode, latents only."""
    posterior_mean = vae.encode(video).latent_dist.mean
    norm_mean = vae.latents_mean.view(1, -1, 1, 1, 1).to(posterior_mean.device, posterior_mean.dtype)
    norm_std = vae.latents_std.view(1, -1, 1, 1, 1).to(posterior_mean.device, posterior_mean.dtype)
    sf = float(getattr(vae.config, "scaling_factor", 1.0))
    return (posterior_mean - norm_mean) * sf / norm_std


def _decode(vae, latents: torch.Tensor, default_temb: float) -> torch.Tensor:
    """VaeTrainer._decode."""
    mean = vae.latents_mean.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    std = vae.latents_std.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    sf = float(getattr(vae.config, "scaling_factor", 1.0))
    z = latents * std / sf + mean
    temb = torch.full((z.shape[0],), default_temb, device=z.device, dtype=z.dtype)
    return vae.decode(z, temb=temb, return_dict=True).sample


@app.command()
def main(
    config_path: str = typer.Argument(..., help="The run's VaeTrainerConfig YAML (model/adapter/normalization are read from it)"),
    checkpoint: str = typer.Option(None, help="Finetuned VAE checkpoint; omit only for the freshly-inflated reference"),
    test_h5: str = typer.Option(..., help="Standalone test.h5, e.g. $SCRATCH/windinet/euler_mq_dataset/256x256_ds/test.h5"),
    num_samples: int = typer.Option(0, help="Evaluate only the first N test sims (0 = all)"),
    batch_size: int = typer.Option(4, help="Sims per forward pass (does not affect the metrics)"),
    output: str = typer.Option("vae_test_eval.json", help="Where to write the JSON report"),
    save_samples: int = typer.Option(3, help="Save fields of N gamma-spread test sims for figures (0 = none)"),
    sample_frames: list[int] = typer.Option([0, 25, 50, 75, 100], help="Frames saved per sample (repeat the option)"),
    samples_dir: str = typer.Option(
        None,
        help="Where the sample .npz files go (default: $SCRATCH/windinet/figure_data/vae_test/<run>, "
        "<run> = the output JSON's parent folder name; <output dir>/samples without $SCRATCH)",
    ),
) -> None:
    with open(config_path) as f:
        cfg = VaeTrainerConfig(**yaml.safe_load(f))
    device = get_default_device()
    channel_order = list(cfg.data.channel_order)
    console.print(f"Device: {device}  config: {config_path}")

    dataset = ShockWaveDataset(Path(test_h5), num_sim_frames=cfg.data.num_sim_frames)
    indices = list(range(len(dataset)))
    if num_samples > 0:
        indices = indices[:num_samples]
    console.print(f"Test set: {test_h5} -- evaluating {len(indices)} of {len(dataset)} sims")
    sample_ids = pick_gamma_spread_ids([dataset.ids[i] for i in indices], save_samples)
    if samples_dir is None:
        run_name = Path(output).resolve().parent.name
        scratch = os.environ.get("SCRATCH")
        samples_dir = (
            str(Path(scratch) / "windinet" / "figure_data" / "vae_test" / run_name)
            if scratch
            else str(Path(output).parent / "samples")
        )
    if sample_ids:
        console.print(f"Saving figure samples {sorted(sample_ids)} (frames {sample_frames}) to {samples_dir}")
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.data.num_dataloader_workers,
    )

    vae = _build_vae(cfg, checkpoint, device)
    autocast_dtype = _AUTOCAST_DTYPES.get(cfg.acceleration.mixed_precision_mode or "no")

    per_sim: list[dict] = []
    t0 = time.time()
    with torch.no_grad():
        for batch in loader:
            orig_F = batch["density"].shape[1]
            x = build_shockwave_video(
                batch,
                device=device,
                channel_mean=cfg.data.channel_mean,
                channel_std=cfg.data.channel_std,
                normalization_clip=cfg.data.normalization_clip,
                channel_order=channel_order,
                log_transform_channels=cfg.data.log_transform_channels,
            )
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
                recon = _decode(vae, _encode(vae, x), cfg.adapter.default_temb)
            recon = recon.float()[:, :, :orig_F]
            target = x[:, :, :orig_F]
            ids = batch.get("id", [None] * recon.shape[0])
            for i in range(recon.shape[0]):
                pred_i, target_i = recon[i : i + 1], target[i : i + 1]
                per_channel = vrmse_per_channel(pred_i, target_i)
                row = {
                    "id": ids[i],
                    "vrmse": vrmse(pred_i, target_i),
                    "vrmse_chmean": float(per_channel.mean().item()),
                }
                row.update({f"vrmse_{name}": float(v) for name, v in zip(channel_order, per_channel.tolist())})
                per_sim.append(row)
                if ids[i] in sample_ids:
                    frames = [f for f in sample_frames if f < orig_F]
                    gt_raw = torch.stack([batch[name][i] for name in channel_order])  # [C, F, H, W]
                    Path(samples_dir).mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(
                        Path(samples_dir) / f"{ids[i]}.npz",
                        frames=np.array(frames),
                        channel_order=np.array(channel_order),
                        gamma=float(batch["meta"]["gamma"][i]),
                        gt_raw=gt_raw[:, frames].numpy().astype(np.float16),
                        gt_norm=target[i][:, frames].cpu().numpy().astype(np.float16),
                        recon_norm=recon[i][:, frames].cpu().numpy().astype(np.float16),
                        channel_mean=np.array(cfg.data.channel_mean),
                        channel_std=np.array(cfg.data.channel_std),
                        normalization_clip=cfg.data.normalization_clip,
                        vrmse_chmean=row["vrmse_chmean"],
                    )
            console.print(f"  {len(per_sim)}/{len(indices)} sims  ({time.time() - t0:.0f}s)")

    keys = ["vrmse", "vrmse_chmean"] + [f"vrmse_{name}" for name in channel_order]
    summary = {}
    for key in keys:
        values = torch.tensor([row[key] for row in per_sim], dtype=torch.float64)
        summary[key] = values.mean().item()
        summary[f"{key}_sem"] = (values.std(unbiased=True) / len(values) ** 0.5).item() if len(values) > 1 else 0.0

    report = {
        "config": config_path,
        "checkpoint": checkpoint,
        "test_h5": test_h5,
        "num_sims": len(per_sim),
        "mixed_precision": cfg.acceleration.mixed_precision_mode,
        "normalization_stats_file": str(cfg.data.normalization_stats_file),
        "channel_order": channel_order,
        "figure_samples": {"ids": sorted(sample_ids), "frames": sample_frames, "dir": samples_dir},
        "metrics": summary,
        "per_sim": per_sim,
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2))

    table = Table(title=f"Test-set reconstruction ({len(per_sim)} sims)")
    table.add_column("metric")
    table.add_column("mean", justify="right")
    table.add_column("sem", justify="right")
    for key in keys:
        table.add_row(key, f"{summary[key]:.4f}", f"{summary[key + '_sem']:.4f}")
    console.print(table)
    console.print(f"Report written to {output}")


if __name__ == "__main__":
    app()
