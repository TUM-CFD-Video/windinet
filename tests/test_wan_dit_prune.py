import torch

from windinet.wan.dit.config import PruneConfig
from windinet.wan.dit.distill import feature_loss, matching_blocks
from windinet.wan.dit.prune import Gap, calibrate, choose, compose, importance, shrink


def tiny(layers: int = 4):
    from diffusers import WanTransformer3DModel

    torch.manual_seed(0)
    return WanTransformer3DModel(
        patch_size=(1, 2, 2),
        num_attention_heads=4,
        attention_head_dim=8,
        in_channels=4,
        out_channels=4,
        text_dim=12,
        freq_dim=16,
        ffn_dim=48,
        num_layers=layers,
        cross_attn_norm=True,
        qk_norm="rms_norm_across_heads",
        rope_max_seq_len=64,
    ).eval()


def batches(n: int = 4) -> list:
    g = torch.Generator().manual_seed(1)
    return [
        (torch.randn(2, 4, 3, 8, 8, generator=g), torch.rand(2, generator=g) * 1000, torch.randn(2, 3, 12, generator=g))
        for _ in range(n)
    ]


def run(net, batch):
    return net(*batch, return_dict=False)[0]


def pruned(cfg: PruneConfig, refit: bool):
    net, data = tiny(), batches()
    scores = importance(net, cfg, calibrate(net, cfg, lambda b: run(net, b), data))
    spec = choose(net, cfg, scores)
    ref = torch.cat([run(net, b) for b in data])
    with torch.no_grad():
        shrink(net, spec | {"bias": spec["bias"] and refit}, scores, refit)
        err = (torch.cat([run(net, b) for b in data]) - ref).square().mean() / ref.square().mean()
    return net, spec, err.item()


def test_refit_brings_every_cut_closer_to_the_unpruned_model():
    for cfg in (
        PruneConfig(heads=2, cross_heads=1, ffn=16),
        PruneConfig(hidden=20),
        PruneConfig(depth=2, blocks="scan"),
    ):
        _, _, plain = pruned(cfg, refit=False)
        net, spec, refit = pruned(cfg, refit=True)
        assert refit < plain, (cfg, refit, plain)


def test_shapes_from_a_spec_load_the_pruned_weights():
    net, spec, _ = pruned(PruneConfig(depth=3, heads=2, cross_heads=1, ffn=16), refit=True)
    assert len(net.blocks) == 3 and any(isinstance(b, Gap) for b in net.blocks)
    fresh = tiny()
    shrink(fresh, spec)
    fresh.load_state_dict(net.state_dict())
    batch = batches(1)[0]
    with torch.no_grad():
        assert torch.equal(run(fresh, batch), run(net, batch))


def test_spread_keeps_first_and_last_and_specs_compose():
    net = tiny(30)
    spec = choose(net, PruneConfig(depth=14), {})
    assert spec["blocks"][0] == 0 and spec["blocks"][-1] == 29 and len(spec["blocks"]) == 14
    first = {"blocks": [0, 2, 4, 6], "hidden": [1, 3, 5], "heads": 3, "cross_heads": None, "ffn": None, "bias": True}
    second = {"blocks": [0, 3], "hidden": [0, 2], "heads": None, "cross_heads": 1, "ffn": None, "bias": True}
    both = compose(first, second)
    assert both["blocks"] == [0, 6] and both["hidden"] == [1, 5] and both["heads"] == 3 and both["cross_heads"] == 1
    assert matching_blocks(both, 8) == [5, 7] and matching_blocks(None, 3) == [0, 1, 2]


def test_feature_loss_is_zero_on_the_kept_slice_of_the_teacher():
    t = torch.randn(2, 5, 6)
    assert feature_loss([t[..., [0, 2]]], [t], [0, 2]) == 0
    assert torch.isclose(feature_loss([torch.zeros(2, 5, 6)], [t], None), torch.tensor(1.0))


def test_grams_stay_fp32_under_bf16_autocast():
    net, cfg = tiny(), PruneConfig(ffn=16)
    with torch.autocast("cpu", torch.bfloat16):
        stats = calibrate(net, cfg, lambda b: run(net, b), batches(2))
    assert all(g.dtype == torch.float32 for g in stats["grams"].values())
