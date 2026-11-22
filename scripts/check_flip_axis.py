"""Check which array axis is y before trusting the x-axis flip augmentation (shockwave_data.flip_y).

flip_y reverses the fields along H (dim -2) and negates the y velocity/momentum. That is
only a symmetry if H is the axis the y velocity points along and the shock travels along
the other one. Both follow from the momentum equation, rho dv/dt ~ -grad p (the convective
term is ignored here, so expect correlations clearly away from 0, not near 1):

  corr(d vY/dt, -(1/rho) dp/dH) should be high, corr(d vY/dt, -(1/rho) dp/dW) near 0,
  and the other way round for vX.

The same correlation for the mirrored sample must keep its sign (if flip_y negated the
wrong channel or reversed the wrong axis it turns negative). mean(vY) should be ~0 and
|mean(vX)| large if the shock runs along W.

SRBA only (needs velocities and density). Login node, a few seconds:
  PYTHONPATH=. python scripts/check_flip_axis.py /e/project1/e-dev-2026d09-262/neptuna_datasets/2D_SRBA_OOOO/256x256/train.h5
"""

import sys

import torch

from windinet.training.shockwave_data import ShockWaveDataset, flip_y


def corr(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten() - a.mean(), b.flatten() - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-12))


def momentum_corrs(s: dict) -> dict[str, float]:
    p, vx, vy = s["pressure"], s["velocityX"], s["velocityY"]  # [T, H, W]
    rho = s["log_density"].exp()
    dp_dh, dp_dw = torch.gradient(p, dim=(1, 2))
    dvx_dt = torch.gradient(vx, dim=0)[0]
    dvy_dt = torch.gradient(vy, dim=0)[0]
    return {
        "vY~H": corr(dvy_dt, -dp_dh / rho), "vY~W": corr(dvy_dt, -dp_dw / rho),
        "vX~W": corr(dvx_dt, -dp_dw / rho), "vX~H": corr(dvx_dt, -dp_dh / rho),
        "mean_vX": float(vx.mean()), "mean_vY": float(vy.mean()),
    }


def main(h5_path: str, n: int = 4) -> None:
    ds = ShockWaveDataset(h5_path, num_sim_frames=49, dataset="srba", frame_stride=2)
    ok = True
    for idx in torch.linspace(0, len(ds) - 1, n).long().tolist():
        s = ds[idx]
        for tag, sample in (("stored ", s), ("flipped", flip_y(s, "srba"))):
            c = momentum_corrs(sample)
            print(f"{s['id'][:40]:40s} {tag} " + "  ".join(f"{k} {v:+.3f}" for k, v in c.items()))
            ok &= c["vY~H"] > abs(c["vY~W"]) and c["vX~W"] > abs(c["vX~H"])
    print("expect: vY~H and vX~W clearly positive for both rows, vY~W and vX~H ~0, mean_vY ~0")
    print("PASS: H is y, velocityY is the H component" if ok else "FAIL: flip_y does not match this data's axes")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main(sys.argv[1])
