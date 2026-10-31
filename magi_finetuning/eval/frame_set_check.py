"""Same rollouts, scored on different frame sets: how much of a benchmark difference is the frame set, not the model.

For ~N test sims (spread over gammas, plus 0064 and 0158, whose late frames are almost constant) the DiT rollout
is generated once (same seeds and sampler as evaluate.py) and scored, with the VAE round trip, on several frame
sets with Harish's benchmark metrics, plus what lies behind them: MLW split into its spatial part and its
beta x temporal part (with the number of temporal wavelet levels the frame count allows), and the target's
variance over the scored frames (VRMSE's denominator). The rollouts are saved (fp16) for later checks.

    sbatch --time=00:45:00 jobs/jupiter/magi1/quick_test.sbatch -m magi_finetuning.eval.frame_set_check \
        --dit-ckpt <ckpt> --vae-ckpt <vae> --sims 20 --out <dir>
"""

import argparse
import json
from pathlib import Path

import pywt
import torch

from magi_finetuning.data import Normalization, stratified_subset
from magi_finetuning.eval.evaluate import SEED, load_physical_fields, predict
from magi_finetuning.eval.report import benchmark_metrics
from magi_finetuning.settings import SAMPLER_WINDOW, TEST_DATA
from windinet.losses import SSIMLoss
from windinet.losses.mlw import _wavelet_loss_spatial, _wavelet_loss_temporal
from windinet.training.shockwave_data import CHANNEL_NAMES, ShockWaveDataset, parse_gamma

FRAME_SETS = {
    "every frame 40..90 (our old set)": list(range(40, 91)),
    "every 10th 40..90": list(range(40, 91, 10)),
    "every frame 10..90": list(range(10, 91)),
    "every 10th 10..90 (Harish)": list(range(10, 91, 10)),
}
NEAR_CONSTANT = ["0064_gamma1.2200000286", "0158_gamma1.3300000429"]
COLUMNS = ["rmse", "vrmse", "ssim", "mlw", "mlw_spatial", "mlw_temporal_x10", "temporal_levels", "target_var"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dit-ckpt", required=True)
    p.add_argument("--vae-ckpt", required=True)
    p.add_argument("--test-data", default=TEST_DATA)
    p.add_argument("--sims", type=int, default=20, help="spread over gammas; 0064 and 0158 are always added")
    p.add_argument("--num-steps", type=int, default=64)
    p.add_argument("--shift", type=float, default=3.0)
    p.add_argument("--window", type=int, default=SAMPLER_WINDOW)
    p.add_argument("--out", required=True)
    return p.parse_args()


def scores_on(frames, prediction, truth, ssim_fn) -> dict:
    """prediction: frames 1..F; truth: frames 0..F (normalized). Harish's metrics + their parts on `frames`."""
    index = torch.tensor(frames, device=truth.device)
    target = truth.index_select(2, index).clamp(-1, 1)
    pred = prediction.index_select(2, index - 1).clamp(-1, 1)
    result = benchmark_metrics(pred, target, ssim_fn)
    result["mlw_spatial"] = float(_wavelet_loss_spatial(pred, target))
    result["mlw_temporal_x10"] = 10 * float(_wavelet_loss_temporal(pred, target))  # mlw_loss: beta = 10
    result["temporal_levels"] = pywt.dwt_max_level(len(frames), pywt.Wavelet("db2").dec_len)
    result["target_var"] = target.var(dim=(0, 2, 3, 4), unbiased=False).mean().item()  # mean over the fields
    return result


def summary_lines(rows, title) -> list[str]:
    lines = ["", f"## {title} ({len(rows)} sims)", "",
             "| frame set | model | " + " | ".join(COLUMNS) + " |", "|---|---|" + "---|" * len(COLUMNS)]
    for name in FRAME_SETS:
        for model in ("DiT", "VAE"):
            values = [r["scores"][name][model] for r in rows]
            means = [sum(v[c] for v in values) / len(values) for c in COLUMNS]
            lines.append(f"| {name} | {model} | " + " | ".join(f"{m:.4g}" for m in means) + " |")
    return lines


def main():
    args = parse_args()
    from magi_finetuning.dit import load_trained_dit, rollout  # GPU imports only where needed
    from magi_finetuning.utils.runtime import init_magi_parallel_state
    from magi_finetuning.vae import load_vae

    out = Path(args.out)
    (out / "rollouts").mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(0)
    init_magi_parallel_state()
    vae = load_vae(args.vae_ckpt)
    normalization = Normalization.of_vae_checkpoint(args.vae_ckpt)
    model, magi_config, is_full_finetune = load_trained_dit(args.dit_ckpt)
    ssim_fn = SSIMLoss(channels=len(CHANNEL_NAMES)).cuda()
    test_set = ShockWaveDataset(args.test_data, num_sim_frames=101)
    sim_ids = sorted(set(stratified_subset(test_set.ids, args.sims)) | set(NEAR_CONSTANT))

    rows = []
    for count, sim_id in enumerate(sim_ids, 1):
        torch.manual_seed(SEED + test_set.ids.index(sim_id))  # the same rollout as evaluate.py
        truth = normalization.apply(load_physical_fields(test_set, sim_id))
        vae_round_trip, generated, _ = predict(vae, model, magi_config, truth, parse_gamma(sim_id), args,
                                            is_full_finetune, rollout)
        torch.save({"generated": generated.half().cpu(), "vae_round_trip": vae_round_trip.half().cpu()},
                   out / "rollouts" / f"{sim_id}.pt")  # normalized frames 1..100
        scores = {name: {"DiT": scores_on(frames, generated, truth, ssim_fn),
                         "VAE": scores_on(frames, vae_round_trip, truth, ssim_fn)}
                  for name, frames in FRAME_SETS.items()}
        rows.append({"sim_id": sim_id, "scores": scores})
        print(f"[{count}/{len(sim_ids)}] {sim_id}: DiT vrmse "
              + ", ".join(f"{name.split(' (')[0]} {s['DiT']['vrmse']:.3f}" for name, s in scores.items()), flush=True)

    (out / "scores.json").write_text(json.dumps(rows))
    lines = ["# Same rollouts, different frame sets", "",
             "Harish's metrics (ssim = 1 - SSIM; mlw = mlw_spatial + mlw_temporal_x10). temporal_levels: wavelet",
             "levels along time for that many frames (db2, maximum level). target_var: variance of the true field",
             "over the scored frames and pixels, mean over the 4 fields (VRMSE's denominator)."]
    lines += summary_lines(rows, "all sims")
    lines += summary_lines([r for r in rows if r["sim_id"] not in NEAR_CONSTANT], "without 0064 and 0158")
    lines += summary_lines([r for r in rows if r["sim_id"] in NEAR_CONSTANT], "only 0064 and 0158")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
