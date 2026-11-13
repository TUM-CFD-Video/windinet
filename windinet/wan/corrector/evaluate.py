"""Evaluation of the corrector: whole autoregressive rollouts against the simulations, beside Wan alone."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from windinet.data.specs import EULERMQ
from windinet.experiment.metrics import Aggregate
from windinet.wan.corrector.model import Corrector, rollout


@torch.no_grad()
def evaluate(model: Corrector, loader: DataLoader, stride: int) -> tuple[dict, dict]:
    """Corrected rollouts and Wan alone (wan_*) from the true frame 0, frame 0 unscored; gain = Wan's VRMSE - the
    corrector's. Also the examples {gamma: (sim id, truth, pred)} of the lowest and the highest gamma this rank saw."""
    device = model.std.device
    corrected, wan_only = Aggregate(EULERMQ.fields, device, given=1), Aggregate(EULERMQ.fields, device, given=1)
    ids, seen, examples = loader.dataset.ids, 0, {}
    ends = {f(round(EULERMQ.scalars(s)["gamma"], 3) for s in ids) for f in (min, max)}
    for wan, truth, gamma in loader:
        wan, truth, gamma = wan.to(device).float(), truth.to(device), gamma.to(device).float()
        pred = rollout(model, wan, truth[:, :1], gamma, range(1, wan.shape[1]))
        for sid, p, w, x in zip(ids[seen:], pred, wan, truth, strict=False):
            corrected.add(p, x)
            wan_only.add(w, x)
            g = round(EULERMQ.scalars(sid)["gamma"], 3)
            if g in ends and g not in examples:
                examples[g] = (sid, x.cpu(), p.cpu())
        seen += len(wan)
    for aggregate in (corrected, wan_only):
        aggregate.reduce()
    metrics = corrected.metrics(stride)
    metrics |= {f"wan_{k}": v for k, v in wan_only.metrics(stride).items()}
    return metrics | {"gain": metrics["wan_vrmse_mean"] - metrics["vrmse_mean"]}, examples
