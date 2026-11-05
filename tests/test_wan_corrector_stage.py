import pytest
import torch
from torch import nn

from windinet.eulermq.data import FIELDS
from windinet.wan.corrector.config import CorrectorModelConfig
from windinet.wan.corrector.data import Windows
from windinet.wan.corrector.model import Corrector, rollout, teacher_forced


@pytest.mark.parametrize("history,first_frame", [(0, False), (1, False), (4, True)])
def test_corrector_starts_as_wan_with_replicate_padding(history, first_frame):
    stats = {f: {"mean": 1.0, "std": 2.0} for f in FIELDS}
    cfg = CorrectorModelConfig(channels=(32, 32), layers_per_block=1, history=history, first_frame=first_frame)
    model = Corrector(cfg, stats)
    wan, gamma = torch.randn(2, 5, 4, 16, 16), torch.tensor([1.13, 1.76])
    assert torch.equal(rollout(model, wan, wan[:, :1], gamma, range(1, 5))[:, 1:], wan[:, 1:])
    assert torch.equal(teacher_forced(model, wan, wan, gamma, range(1, 5), 0.1), wan[:, 1:])
    convs = [m for m in model.unet.modules() if isinstance(m, nn.Conv2d) and m.padding != (0, 0)]
    assert convs and all(m.padding_mode == "replicate" for m in convs)


def test_windows_share_one_end():
    batch = next(iter(Windows(sims=10, batch=4, window=4, frames=49, seed=0)))
    assert len(batch) == 4 and len({end for _, end in batch}) == 1 and 5 <= batch[0][1] <= 49
