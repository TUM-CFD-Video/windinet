"""Standardized rollout videos (docs/video_spec.md) from eval_dit_vrmse.py --save_rollouts.

Every <rollout_dir>/<sid>.npz (gt, pred, pred_vae_only [T, C, H, W] in physical units, times, channels) becomes
<out_dir>/<model>_<sid>.mp4 (one row per channel: ground truth | prediction | signed error), plus with --vae_only
<out_dir>/<model>-vae_only_<sid>.mp4 for the VAE reconstruction. The scales come only from the shared training
statistics (--stats, written by compute_channel_stats.py --video-stats), never from the rollouts.

    python scripts/make_rollout_videos.py outputs/eval_dit_vrmse/rollouts \\
        configs/video/euler_mq_256_video_stats.json videos/cond9 --model cond9 --json
"""

import argparse
import json
from pathlib import Path

import numpy as np

from windinet.video import write_frame, write_video


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rollout_dir", type=Path, help="eval_dit_vrmse.py --save_rollouts output (<sid>.npz)")
    ap.add_argument("stats", type=Path, help="shared video statistics json {channel: Stats}")
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--model", required=True, help="model name in the file names and the title, e.g. cond9")
    ap.add_argument("--sim_ids", nargs="+", default=None, help="only these sims (default: every npz)")
    ap.add_argument("--vae_only", action="store_true", help="also a video of the VAE reconstruction")
    ap.add_argument("--slowdown", type=float, default=8.0, help="video seconds per physical second")
    ap.add_argument("--json", action="store_true", help="also <video>.json with per-frame RMSE and fractions")
    ap.add_argument("--per_channel", action="store_true", help="one video per channel instead of one row each")
    ap.add_argument("--frames", type=int, nargs="*", default=None,
                    help="also a png of these time steps (e.g. 25 50 75 96)")
    args = ap.parse_args()

    all_stats = json.loads(args.stats.read_text())
    paths = sorted(args.rollout_dir.glob("*.npz"))
    if args.sim_ids:
        paths = [args.rollout_dir / f"{sid}.npz" for sid in args.sim_ids]
    if not paths:
        raise SystemExit(f"no rollouts in {args.rollout_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for path in paths:
        data = np.load(path)
        sid = str(data["sim_id"]) if "sim_id" in data else path.stem
        names = [str(c) for c in data["channels"]]
        missing = [c for c in names if c not in all_stats]
        if missing:
            raise SystemExit(f"{args.stats} has no statistics for {missing}")
        n_cond = int(data["num_conditioning_frames"]) if "num_conditioning_frames" in data else 1
        t_cond = float(data["times"][n_cond - 1])
        labels = [("pred", args.model)] + ([("pred_vae_only", f"{args.model}-vae_only")] if args.vae_only else [])
        for key, model in labels:
            # The DiT is given frames 0..n_cond-1: say so, its error there is not a prediction.
            given = f" (conditioned on GT t ≤ {t_cond:.3g} s)" if key == "pred" else ""
            title = f"{model}, trajectory {sid}{given}"
            groups = [[i] for i in range(len(names))] if args.per_channel else [list(range(len(names)))]
            for idx in groups:
                sel = [names[i] for i in idx]
                stats = [all_stats[c] for c in sel]
                stem = f"{model}_{sid}" + (f"_{sel[0]}" if args.per_channel else "")
                out = args.out_dir / stem.replace(".", "p")  # video.py uses with_suffix: a "." would cut the name
                gt, pred = data["gt"][:, idx], data[key][:, idx]
                frames = write_video(gt, pred, data["times"], sel, stats, out, title, args.slowdown, args.json)
                rmse = np.array([[f[c]["rmse"] if f[c]["rmse"] is not None else np.nan for c in sel] for f in frames])
                print(f"{out}.mp4  {len(frames)} frames, mean RMSE "
                      + "  ".join(f"{c} {v:.4g}" for c, v in zip(sel, np.nanmean(rmse, axis=0))))
                for t in args.frames or []:
                    write_frame(gt[t], pred[t], float(data["times"][t]), sel, stats,
                                out.with_name(f"{out.name}_t{t}.png"), title)


if __name__ == "__main__":
    main()
