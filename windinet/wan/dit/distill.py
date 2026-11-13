"""Distillation from a frozen teacher: block outputs captured by hooks, the feature loss between student and teacher,
and the closeness of their velocities on a fixed batch."""

from __future__ import annotations

import torch
from torch import nn


class Features:
    """Keeps the output of the blocks `indices` of `transformer` from its last forward."""

    def __init__(self, transformer: nn.Module, indices: list[int]):
        self.indices, self.out = indices, {}
        for i in indices:
            transformer.blocks[i].register_forward_hook(self._keep(i))

    def _keep(self, i: int):
        def hook(module, args, out):
            self.out[i] = out

        return hook

    def __call__(self) -> list[torch.Tensor]:
        return [self.out[i] for i in self.indices]


def matching_blocks(spec: dict | None, depth: int) -> list[int]:
    """The teacher block whose output each student block stands for: the last of its group of dropped blocks."""
    kept = spec["blocks"] if spec else list(range(depth))
    return [nxt - 1 for nxt in kept[1:]] + [depth - 1]


def feature_loss(student: list[torch.Tensor], teacher: list[torch.Tensor], hidden: list[int] | None) -> torch.Tensor:
    """Mean over blocks of ‖s - t‖² / ‖t‖², the teacher on the kept stream channels when the stream is pruned."""
    losses = []
    for s, t in zip(student, teacher, strict=True):
        t = (t[..., hidden] if hidden is not None else t).float()
        losses.append((s.float() - t).square().sum() / t.square().sum())
    return torch.stack(losses).mean()


def closeness(v: torch.Tensor, v_teacher: torch.Tensor) -> dict:
    """Relative MSE and mean cosine (per sample) of the student's velocity against the teacher's."""
    rel = ((v - v_teacher).square().sum() / v_teacher.square().sum()).item()
    cos = nn.functional.cosine_similarity(v.flatten(1), v_teacher.flatten(1)).mean().item()
    return {"teacher_rel_mse": rel, "teacher_cos": cos}


class Teacher:
    """The frozen teacher with its own conditioning; with `features`, hooks on the student's blocks and on the
    teacher's matching ones (`spec`: the student's prune spec, None = same depth and width)."""

    def __init__(self, dit, cond: nn.Module, student, spec: dict | None, features: bool):
        self.dit, self.cond, self.hidden = dit, cond, spec["hidden"] if spec else None
        if features:
            self.mine = Features(student.transformer, list(range(len(student.transformer.blocks))))
            self.its = Features(dit.transformer, matching_blocks(spec, len(dit.transformer.blocks)))

    @torch.no_grad()
    def velocity(self, x: torch.Tensor, sigma: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        return self.dit.velocity(x, sigma * 1000, self.cond(gamma))

    def feature_loss(self) -> torch.Tensor:
        """Of the last student and teacher forwards."""
        return feature_loss(self.mine(), self.its(), self.hidden)
