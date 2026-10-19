import json

import torch

from windinet.eulermq.data import FIELDS
from windinet.experiment.runs import settings
from windinet.wan.dit.conditioning import GammaTokens
from windinet.wan.dit.flow import draw_sigmas, noised, velocity_target
from windinet.wan.dit.latents import LatentDataset


def test_flow_endpoints_and_sigma_range():
    x0, noise = torch.randn(3, 2, 2, 4, 4), torch.randn(3, 2, 2, 4, 4)
    assert torch.equal(noised(x0, noise, torch.zeros(3)), x0)
    assert torch.equal(noised(x0, noise, torch.ones(3)), noise)
    assert torch.equal(velocity_target(x0, noise), noise - x0)
    torch.manual_seed(0)
    plain, shifted = draw_sigmas(1000, "cpu"), draw_sigmas(1000, "cpu", shift=3.0)
    assert plain.min() > 0 and plain.max() < 1 and shifted.mean() > plain.mean()  # the shift pushes towards noise


def test_latent_dataset_reads_files_and_truncates(tmp_path):
    (tmp_path / "val").mkdir()
    for sid in ("b_gamma1.4", "a_gamma1.13"):
        torch.save(
            {"latents": torch.randn(1, 16, 5, 2, 2), "gamma": float(sid.split("gamma")[1])},
            tmp_path / "val" / f"{sid}.pt",
        )
    data = LatentDataset(tmp_path, "val", frames=2)
    assert data.ids == ["a_gamma1.13", "b_gamma1.4"] and data[1][0].shape == (1, 16, 2, 2, 2) and data[1][1] == 1.4


def test_gamma_tokens_shape_and_uncond():
    cond = GammaTokens(tokens=4, gamma_range=(1.0, 2.0))
    gamma = torch.tensor([1.13, 1.76])
    assert cond(gamma).shape == (2, 4, 4096) and torch.equal(cond.uncond(gamma), torch.zeros(2, 4, 4096))


def test_settings_phrase_for_a_dit_run():
    cfg = {
        "cond": {"kind": "gamma", "first_frame": True},
        "train": {"steps": 8000, "lr": 1e-5, "batch_sims": 32},
        "sampling": {"steps": 20},
        "vae": "results/wan/vae/rgb3_two_stage",
    }
    assert settings(cfg) == (
        "gamma tokens, clean first frame, from pretrained Wan 2.1, 8000 steps, lr 1e-05, batch 32, "
        "20-step rollouts, decoders of rgb3_two_stage"
    )


def test_benchmark_scores_the_cfd_bench_frames_the_stride_reaches(tmp_path):
    from windinet.wan.dit.benchmark import Benchmark

    stats = {f: {"mean": 0.0, "std": 1.0} for f in FIELDS}
    (tmp_path / "stats.json").write_text(json.dumps({"data_normalization_stats": stats}))
    x = torch.rand(49, 4, 16, 16)  # 101 frames at stride 2: frames 0, 2, ..., 96
    bench = Benchmark(tmp_path / "stats.json", list(FIELDS), 2, "cpu")
    bench.add(x, x)
    m = bench.metrics()
    assert m["bench_frames"] == [40, 50, 60, 70, 80, 90]
    assert m["bench_rmse"] < 1e-3 and m["bench_vrmse"] < 1e-3 and m["bench_ssim"] < 1e-3 and m["bench_mlw"] < 1e-3
    bench.add(x, torch.zeros_like(x))
    assert bench.metrics()["bench_vrmse"] > 0.4  # a flat prediction scores about 1 on this sim, averaged with the 0
