"""Where does the error sit: shocks, contact discontinuities or smooth flow,
and which latent tokens carry it (eval_dit_vrmse.py --shock_analysis).

Regions, per frame, from the ground truth in physical units, shared by every
channel (so the per-channel numbers are comparable):
  - shock:      top `top_frac` of |grad log p| = |grad p| / p. Pressure jumps
                across a shock but is continuous across a contact
                discontinuity / slip line, so this picks shocks only. The
                relative gradient keeps weak, far shocks visible next to
                strong ones (pressure spans orders of magnitude over gamma
                and time).
  - shock_band: the shock dilated by `dilate_px`, minus the shock. Takes the
                error of a shock predicted a few pixels off.
  - contact:    top `top_frac` of |grad log rho| (srba: log_density is
                already the log), dilated by `dilate_px`, outside the dilated
                shock. Density jumps at shocks and contacts; with the shocks
                removed this is contacts / material interfaces (the srba
                bubble walls).
  - smooth:     everything else.
Momentum / velocity jump at both kinds of discontinuity, so they do not pick
the regions; their error is still reported per region.

top_frac labels that fraction of every frame, so a frame without a shock
still gets a "shock" region; the scored frames (40..90) all have one.

Tokens: the LTX VAE packs frame 0 alone into latent frame 0, then 8 frames
per latent frame, and 32x32 pixels per latent position, so token (t, i, j)
nominally covers frames token_frame_range(t) and that 32x32 patch. Nominally:
the decoder's receptive field is wider than the patch. Latent channels mix
all physical channels, so the per-channel split exists in pixel space only.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

REGIONS = ("shock", "shock_band", "contact", "smooth")
LOG_EPS = 1e-12
# Token shock-fraction bins for the summary: no shock pixel, then increasing.
SHOCK_FRAC_BINS = (0.0, 0.02, 0.05, 0.1, 0.2)


def _grad_mag(x: Tensor) -> Tensor:
    """|grad x| over the last two dims, central differences, grid spacing 1."""
    gy, gx = torch.gradient(x, dim=(-2, -1))
    return torch.sqrt(gx**2 + gy**2)


def _top_frac(score: Tensor, frac: float) -> Tensor:
    """[F, H, W] -> bool, the top `frac` of each frame (ties included, zeros never)."""
    flat = score.flatten(1)
    k = max(1, round(frac * flat.shape[1]))
    thr = flat.topk(k, dim=1).values[:, -1].view(-1, 1, 1)
    return (score >= thr) & (score > 0)


def _dilate(mask: Tensor, px: int) -> Tensor:
    if px <= 0:
        return mask
    return F.max_pool2d(mask.float().unsqueeze(1), 2 * px + 1, stride=1, padding=px).squeeze(1).bool()


def region_masks(fields: dict[str, Tensor], top_frac: float = 0.05, dilate_px: int = 3) -> dict[str, Tensor]:
    """Physical ground-truth fields, name -> [F, H, W] -> REGIONS, name -> bool [F, H, W], a partition."""
    shock_score = _grad_mag(torch.log(fields["pressure"].float().clamp(min=LOG_EPS)))
    if "log_density" in fields:
        density_score = _grad_mag(fields["log_density"].float())
    else:
        density_score = _grad_mag(torch.log(fields["density"].float().clamp(min=LOG_EPS)))
    shock = _top_frac(shock_score, top_frac)
    near_shock = _dilate(shock, dilate_px)
    contact = _dilate(_top_frac(density_score, top_frac), dilate_px) & ~near_shock
    return {
        "shock": shock,
        "shock_band": near_shock & ~shock,
        "contact": contact,
        "smooth": ~near_shock & ~contact,
    }


class RegionErrorStats:
    """Squared error per region and channel, summed over pixels of every sim
    added, for several predictions (e.g. vae_only, vae_dit) of one target."""

    def __init__(self, channel_names: list[str], labels: tuple[str, ...]):
        self.channel_names = channel_names
        self.labels = labels
        n_ch = len(channel_names)
        self.se = {label: {r: torch.zeros(n_ch, dtype=torch.float64) for r in REGIONS} for label in labels}
        self.count = {r: 0 for r in REGIONS}

    def add(self, masks: dict[str, Tensor], preds: dict[str, Tensor], target: Tensor) -> None:
        """masks: region -> [F, H, W]; preds / target: [C, F, H, W], same frames."""
        for r in REGIONS:
            self.count[r] += int(masks[r].sum())
        for label in self.labels:
            se = (preds[label].float() - target.float()) ** 2
            for r in REGIONS:
                self.se[label][r] += (se * masks[r]).sum(dim=(1, 2, 3)).double().cpu()

    def summary(self) -> dict:
        """Per label and region: area fraction, RMSE per channel (normalized
        space), share of the channel's total squared error, and their channel
        means. concentration = error share / area share (1 = no more error
        than its size)."""
        total_px = sum(self.count.values())
        out = {"area_frac": {r: self.count[r] / total_px for r in REGIONS}}
        for label in self.labels:
            total_se = sum(self.se[label][r] for r in REGIONS)
            out[label] = {}
            for r in REGIONS:
                n = max(self.count[r], 1)
                rmse = (self.se[label][r] / n).sqrt()
                share = self.se[label][r] / total_se.clamp(min=1e-30)
                out[label][r] = {
                    "rmse": dict(zip(self.channel_names, rmse.tolist())),
                    "rmse_chmean": float(rmse.mean()),
                    "se_share": dict(zip(self.channel_names, share.tolist())),
                    "se_share_chmean": float(share.mean()),
                    "concentration_chmean": float(share.mean()) / max(out["area_frac"][r], 1e-30),
                }
        return out


def token_frame_range(t: int, n_frames: int) -> tuple[int, int]:
    """Frames [start, end) behind latent frame t (frame 0 alone, then 8 per latent)."""
    if t == 0:
        return 0, 1
    return min(8 * t - 7, n_frames), min(8 * t + 1, n_frames)


def token_mean(x: Tensor, n_lat: int, h: int, w: int) -> Tensor:
    """[..., F, H, W] -> [..., n_lat, h, w], the mean over each token's footprint (NaN if it has no frame)."""
    n_frames, H, W = x.shape[-3:]
    if H % h or W % w:
        raise ValueError(f"pixel grid {H}x{W} is not a multiple of the latent grid {h}x{w}")
    out = []
    for t in range(n_lat):
        start, end = token_frame_range(t, n_frames)
        if end <= start:
            out.append(x.new_full((*x.shape[:-3], h, w), float("nan")))
            continue
        chunk = x[..., start:end, :, :].float().mean(dim=-3)
        out.append(chunk.reshape(*chunk.shape[:-2], h, H // h, w, W // w).mean(dim=(-3, -1)))
    return torch.stack(out, dim=-3)


def _rank(x: Tensor) -> Tensor:
    r = torch.empty_like(x)
    r[x.argsort()] = torch.arange(len(x), dtype=x.dtype)
    return r


def _pearson(a: Tensor, b: Tensor) -> float:
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm()).clamp(min=1e-30))


def token_summary(tokens: dict[str, Tensor], first_free_latent: int, top_k: int = 20) -> dict:
    """tokens (stacked over sims): shock_frac / contact_frac / latent_mse
    [N, T, h, w], pixel_mse_vae_dit / pixel_mse_vae_only [N, C, T, h, w].
    Latent frames before first_free_latent are the given condition (no DiT
    error) and are left out."""
    sl = slice(first_free_latent, None)
    shock = tokens["shock_frac"][:, sl].double()
    lat = tokens["latent_mse"][:, sl].double()
    pix_dit = tokens["pixel_mse_vae_dit"][:, :, sl].double().mean(dim=1)
    pix_vae = tokens["pixel_mse_vae_only"][:, :, sl].double().mean(dim=1)
    keep = ~(shock.isnan() | lat.isnan() | pix_dit.isnan())
    s, l, pd, pv = shock[keep], lat[keep], pix_dit[keep], pix_vae[keep]

    bins = []
    edges = list(SHOCK_FRAC_BINS) + [float("inf")]
    for k in range(len(edges) - 1):
        lo, hi = edges[k], edges[k + 1]
        sel = (s == 0) if k == 0 else ((s > lo) & (s <= hi))
        n = int(sel.sum())
        bins.append({
            "shock_frac": "0" if k == 0 else f"({lo}, {hi}]",
            "n_tokens": n,
            "latent_mse": float(l[sel].mean()) if n else None,
            "pixel_mse_vae_dit_chmean": float(pd[sel].mean()) if n else None,
            "pixel_mse_vae_only_chmean": float(pv[sel].mean()) if n else None,
        })

    # Worst tokens of the sim-mean map (shock positions differ per sim, so
    # this is where error piles up on average).
    mean_lat = tokens["latent_mse"][:, sl].double().nanmean(dim=0)
    mean_shock = tokens["shock_frac"][:, sl].double().nanmean(dim=0)
    n_frames = tokens["n_frames"]
    worst = []
    for flat_idx in mean_lat.flatten().argsort(descending=True)[:top_k].tolist():
        t, i, j = torch.unravel_index(torch.tensor(flat_idx), mean_lat.shape)
        lat_t = int(t) + first_free_latent
        start, end = token_frame_range(lat_t, n_frames)
        worst.append({
            "latent_frame": lat_t, "frames": [start, end - 1], "row": int(i), "col": int(j),
            "latent_mse": float(mean_lat[t, i, j]), "shock_frac": float(mean_shock[t, i, j]),
        })

    return {
        "n_tokens": int(keep.sum()),
        "first_free_latent": first_free_latent,
        "latent_mse_vs_shock_frac": {"pearson": _pearson(l, s), "spearman": _pearson(_rank(l), _rank(s))},
        "pixel_mse_vae_dit_vs_shock_frac": {"pearson": _pearson(pd, s), "spearman": _pearson(_rank(pd), _rank(s))},
        "latent_mse_vs_pixel_mse_vae_dit": {"pearson": _pearson(l, pd), "spearman": _pearson(_rank(l), _rank(pd))},
        "by_shock_frac": bins,
        "worst_tokens_sim_mean": worst,
    }


def save_token_heatmaps(tokens: dict[str, Tensor], path: Path, dpi: int = 150) -> None:
    """Sim-mean maps per latent frame: shock fraction, contact fraction, latent MSE, VAE+DiT pixel MSE."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [
        ("shock fraction", tokens["shock_frac"].double().nanmean(dim=0), "viridis"),
        ("contact fraction", tokens["contact_frac"].double().nanmean(dim=0), "viridis"),
        ("latent MSE", tokens["latent_mse"].double().nanmean(dim=0), "magma"),
        ("VAE+DiT pixel MSE", tokens["pixel_mse_vae_dit"].double().mean(dim=1).nanmean(dim=0), "magma"),
    ]
    n_lat = rows[0][1].shape[0]
    fig, axes = plt.subplots(len(rows), n_lat, figsize=(1.4 * n_lat + 1, 1.5 * len(rows)), squeeze=False)
    for r, (name, m, cmap) in enumerate(rows):
        vmax = float(torch.nan_to_num(m).max()) or 1.0
        for t in range(n_lat):
            ax = axes[r, t]
            im = ax.imshow(m[t].numpy(), cmap=cmap, vmin=0, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                start, end = token_frame_range(t, tokens["n_frames"])
                ax.set_title(f"z{t}\nf{start}-{end - 1}", fontsize=7)
            if t == 0:
                ax.set_ylabel(name, fontsize=7)
        fig.colorbar(im, ax=axes[r, :].tolist(), fraction=0.02, pad=0.01)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
