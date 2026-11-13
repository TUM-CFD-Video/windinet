import importlib.util
import json
from pathlib import Path

import pytest
import torch
import typer

from windinet.data.specs import EULERMQ, SRBA_OOOO
from windinet.experiment.config import load_config
from windinet.experiment.metrics import FIELD_METRICS, Aggregate, ensemble_scores, field_metrics, power
from windinet.experiment.optim import EMA, warmup_cosine
from windinet.experiment.runs import settings
from windinet.wan.dit import evaluate
from windinet.wan.dit.conditioning import GammaModulation, ScalarTokens, build
from windinet.wan.dit.config import CondConfig, DitStageConfig
from windinet.wan.dit.flow import clean_estimate, draw_sigmas, frame_sigmas, gated, noised, velocity_target
from windinet.wan.dit.latents import LatentDataset, condition, latent_id, with_augmentations
from windinet.wan.dit.model import Modulation, widen_patchify
from windinet.wan.dit.rollout import rollout
from windinet.wan.figures import save_dit_curves
from windinet.wan.vae.loss import amse, l1_gdl


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
    assert data.ids == ["a_gamma1.13", "b_gamma1.4"] and data[1][0].shape == (1, 16, 2, 2, 2)
    assert torch.equal(data[1][1], torch.tensor([1.4]))  # a file before the scalars: gamma alone
    (tmp_path / "encode.yaml").write_text("config: {data: {augment: true}}\n")
    assert with_augmentations(tmp_path, ["a_gamma1.13"]) == ["a_gamma1.13"] + [f"a_gamma1.13_s{k}" for k in range(1, 8)]


def test_latent_files_keep_the_scalars_and_the_turned_direction(tmp_path):
    sid = "005_Mas1.30_sb1_x"
    assert condition(SRBA_OOOO, sid, 3) == {"scalars": {"mach": 1.3, "bubbles": 1.0}, "direction": (-1.0, 0.0)}
    assert condition(EULERMQ, "0001_gamma1.4") == {"scalars": {"gamma": 1.4}}
    (tmp_path / "val").mkdir()
    item = {"latents": torch.zeros(1, 16, 2, 1, 1), "scalars": {"mach": 1.3, "bubbles": 2.0}, "direction": (1.0, 0.0)}
    torch.save(item, tmp_path / "val" / f"{sid}.pt")
    c = LatentDataset(tmp_path, "val")[0][1]
    assert c.dtype == torch.float32 and torch.equal(c, torch.tensor([1.3, 2.0, 1.0, 0.0]))


def test_one_scalar_keeps_the_gamma_tokens_of_old_runs():
    from windinet.config import ScalarConditioningConfig
    from windinet.scalar_embeddings import ScalarEmbedding

    torch.manual_seed(0)
    cond = ScalarTokens(4, EULERMQ.ranges)
    mlp = [f"embed.mlp.{i}.{p}" for i in (0, 1, 4, 5, 8) for p in ("weight", "bias")]
    assert list(cond.state_dict()) == ["embed.ranges_min", "embed.ranges_max", "embed.fourier.frequencies", *mlp]
    torch.manual_seed(0)
    old = ScalarEmbedding(
        ScalarConditioningConfig(
            enabled=True, scalar_names=["gamma"], scalar_ranges={"gamma": (1.0, 2.0)}, num_tokens_per_scalar=4
        )
    )
    gamma = torch.tensor([[1.13], [1.76]])
    assert torch.equal(cond(gamma), old(gamma))


def test_scalar_tokens_tell_the_inputs_apart():
    torch.manual_seed(0)
    cond = build(CondConfig(), SRBA_OOOO, torch.device("cpu"))
    c = torch.tensor([[1.3, 2.0, 0.0, 1.0], [1.8, 4.0, 1.0, 0.0]])
    out = cond(c)
    assert out.shape == (2, 16, 4096) and torch.equal(cond.uncond(c), torch.zeros(2, 16, 4096))
    assert not torch.allclose(cond(c[:, [0, 1, 3, 2]]), out)  # dir_h and dir_w swapped
    with pytest.raises(ValueError, match="srba_oooo"):
        build(CondConfig(kind="gamma_adaln"), SRBA_OOOO, torch.device("cpu"))


def test_old_dit_run_configs_with_gamma_range_load():
    run = "results/wan/eulermq/dit/res256/pairs/unrolled/unrolled2"
    assert "gamma_range" in load_config(Path(run) / "config.yaml", [])["cond"]
    assert DitStageConfig(**load_config(Path(run) / "config.yaml", [])).cond == CondConfig()
    assert DitStageConfig(**load_config(Path("configs/wan/dit_eval.yaml"), [f"load={run}"])).train.steps == 0


def test_settings_phrase_for_a_dit_run():
    cfg = {
        "cond": {"kind": "gamma", "context": 1},
        "train": {"steps": 8000, "lr": 1e-5, "batch_sims": 32},
        "sampling": {"steps": 20},
        "vae": "results/wan/eulermq/vae/res256/rgb3/two_stage",
    }
    assert settings(cfg) == "8000 steps, lr 1e-05, batch 32"
    assert (
        settings({"load": "results/wan/eulermq/dit/run", "load_step": 4000, "train": {"steps": 0}})
        == "from run @ 4000, eval only"
    )


def test_widen_keeps_latent_zero_and_zeroes_the_rest():
    from diffusers import WanTransformer3DModel

    torch.manual_seed(0)
    model = WanTransformer3DModel(
        num_attention_heads=2, attention_head_dim=8, in_channels=16, text_dim=8, freq_dim=32, ffn_dim=16, num_layers=1
    ).eval()
    x, t, cond = torch.randn(1, 16, 2, 4, 4), torch.tensor([500.0]), torch.randn(1, 3, 8)
    ref = model(x, t, cond, return_dict=False)[0]
    widen_patchify(model, 2)
    out = model(torch.cat([x, torch.randn_like(x)], 1), t, cond, return_dict=False)[0]
    assert out.shape[1] == 32 and torch.allclose(out[:, :16], ref, atol=1e-5) and out[:, 16:].abs().max() == 0


def test_cosine_decays_with_and_without_warmup():
    plain, warm = warmup_cosine(0, 100, 1e-3), warmup_cosine(10, 100, 1e-3)
    assert plain(0) == 1 and abs(plain(50) - 0.5005) < 1e-3 and abs(plain(100) - 1e-3) < 1e-9
    assert abs(warm(0) - 0.01) < 1e-9 and warm(10) == 1 and abs(warm(100) - 1e-3) < 1e-9


def test_config_extends_the_parent_run(tmp_path):
    (tmp_path / "parent").mkdir()
    parent = "name: parent\nload: old\nload_step: 4000\ntrain: {steps: 8000, lr: 1e-5}\n"
    (tmp_path / "parent" / "config.yaml").write_text(parent)
    (tmp_path / "anneal.yaml").write_text("extends: load\nload_step: null\ntrain: {steps: 2000}\n")
    cfg = load_config(tmp_path / "anneal.yaml", [f"load={tmp_path / 'parent'}", "train.lr=3.0e-5"])
    assert cfg["train"] == {"steps": 2000, "lr": 3e-5} and cfg["load_step"] is None and cfg["name"] == "parent"


def test_ema_warmed_up_decay_and_swap():
    params = [torch.nn.Parameter(torch.zeros(3)), torch.nn.Parameter(torch.zeros(2, 2))]
    ema = EMA(params, 0.999)
    for p in params:
        p.data.fill_(1.0)
    ema.update()  # decay min(0.999, 2 / 11): the average moves 9 / 11 of the way
    ema.swap()
    assert all(torch.allclose(p, torch.full_like(p, 9 / 11)) for p in params)
    ema.swap()
    assert all(torch.equal(p, torch.ones_like(p)) for p in params)


def test_curves_of_an_eval_only_run(tmp_path):
    for i, values in enumerate(({"gamma1.13": 0.4, "gamma1.76": 0.5}, {"mach1.3": 0.2, "bubbles2": 0.3})):
        final = {"vrmse_per_frame": [0.0] + [0.3] * 100} | {f"vrmse_mean_{k}": v for k, v in values.items()}
        save_dit_curves([], [], final, ["rho"], tmp_path / f"curves{i}.png")
        assert (tmp_path / f"curves{i}.png").exists()


def test_field_metrics_of_identity_and_of_a_flat_prediction():
    x = torch.randn(3, 2, 16, 16)
    m = dict(zip(FIELD_METRICS, field_metrics(x, x), strict=True))
    ones = ("ssim", "power_low", "power_mid", "power_high")
    assert all(torch.allclose(m[k], torch.ones(2)) for k in ones)
    assert all(v.abs().max() < 1e-3 for k, v in m.items() if k not in ones)
    m = dict(zip(FIELD_METRICS, field_metrics(x.mean((0, 2, 3), keepdim=True).expand_as(x), x), strict=True))
    assert all(torch.allclose(m[k], torch.ones(2), atol=1e-3) for k in ("vrmse", "h1", "spectral_mid", "spectral_high"))
    assert m["ssim"].max() < 0.5 and m["spectral_low"].max() < 1  # the flat field keeps the k = 0 mode
    assert m["power_mid"].max() < 1e-3 and m["power_high"].max() < 1e-3


def test_ensemble_scores_of_a_calibrated_ensemble():
    torch.manual_seed(0)
    n, mean = 3, torch.randn(1, 50, 2, 32, 32)  # few samples: the (n + 1) / n correction matters
    samples, x = mean + torch.randn(n, 50, 2, 32, 32), mean[0] + torch.randn(50, 2, 32, 32)
    spread, skill, crps = ensemble_scores(samples, x).sum(1)
    assert torch.allclose(((n + 1) / n * spread / skill).sqrt(), torch.ones(2), atol=0.03)
    pairs = (samples[:, None] - samples[None]).abs().mean((0, 1)) * n / (2 * (n - 1))  # fair CRPS by brute force
    fair = ((samples - x).abs().mean(0) - pairs).mean((0, 2, 3)) / x.std((0, 2, 3), unbiased=False)
    assert torch.allclose(crps / len(x), fair, atol=1e-4)


def test_power_ratio_sees_blur_but_not_the_mean():
    torch.manual_seed(0)
    x = 1 + torch.randn(4, 2, 64, 64)  # a non-zero mean: k = 0 is left out
    assert torch.allclose(power(1 + 0.5 * (x - 1), x), torch.full((2, 3), 0.25), atol=1e-4)
    k = torch.fft.fftfreq(64).abs()
    smooth = torch.fft.ifft2(torch.fft.fft2(x) * ((k[:, None] < 0.25) & (k[None] < 0.25))).real  # fine scales gone
    assert power(smooth, x)[:, 2].max() < 0.5 and torch.allclose(power(smooth, x)[:, 0], torch.ones(2), atol=1e-4)


def test_gamma_modulation_starts_as_the_pretrained_model():
    from diffusers import WanTransformer3DModel

    torch.manual_seed(0)
    net = WanTransformer3DModel(
        patch_size=(1, 2, 2),
        num_attention_heads=2,
        attention_head_dim=8,
        in_channels=4,
        out_channels=4,
        text_dim=12,
        freq_dim=16,
        ffn_dim=24,
        num_layers=1,
        cross_attn_norm=True,
        qk_norm="rms_norm_across_heads",
        rope_max_seq_len=64,
    ).eval()
    modulation = Modulation()
    net.condition_embedder.time_embedder.register_forward_hook(modulation)
    cond = GammaModulation((1.0, 2.0), net.condition_embedder)
    x, t, tokens = torch.randn(2, 4, 2, 4, 4), torch.tensor([500.0, 20.0]), torch.zeros(2, 1, 12)
    gamma = torch.tensor([[1.13], [1.76]])
    with torch.no_grad():
        plain = net(x, t, tokens, return_dict=False)[0]
        modulation.offset = cond(gamma)  # zero-initialised last layer: no change yet
        assert torch.equal(net(x, t, tokens, return_dict=False)[0], plain)
        modulation.offset = torch.ones(2, 16)
        assert not torch.allclose(net(x, t, tokens, return_dict=False)[0], plain)
        modulation.offset = None
        assert torch.equal(net(x, t, tokens, return_dict=False)[0], plain)


def test_frame_sampler_stride_keeps_4k_plus_1_frames(tmp_path):
    import h5py

    from windinet.data.frames import FrameSampler, symmetry

    with h5py.File(tmp_path / "t.h5", "w") as f:
        for name in EULERMQ.fields:
            f.create_dataset(f"0000_gamma1.3/{name}", data=torch.arange(11.0)[:, None, None, None].expand(11, 1, 2, 2))
    whole = FrameSampler(tmp_path / "t.h5", ["0000_gamma1.3"], None, 0, spec=EULERMQ)[0][0]
    strided = FrameSampler(tmp_path / "t.h5", ["0000_gamma1.3"], None, 0, stride=2, spec=EULERMQ)[0][0]
    assert whole[:, 0, 0, 0].tolist() == list(range(9))  # 11 frames trimmed to 9 = 4 * 2 + 1
    assert strided[:, 0, 0, 0].tolist() == [0, 2, 4, 6, 8]  # 6 strided frames trimmed to 5
    clip = FrameSampler(tmp_path / "t.h5", ["0000_gamma1.3"], 3, 0, consecutive=True, stride=2, spec=EULERMQ)[0][0]
    clip = clip[:, 0, 0, 0]
    assert clip[1] - clip[0] == 2 and clip[2] - clip[1] == 2 and clip[-1] <= 10
    plain = FrameSampler(tmp_path / "t.h5", ["0000_gamma1.3"], None, 0, spec=EULERMQ)[0][0]
    augmented = FrameSampler(tmp_path / "t.h5", ["0000_gamma1.3"], None, 0, augment=True, spec=EULERMQ)[0][0]
    assert any(torch.equal(augmented, symmetry(plain, k, EULERMQ)) for k in range(8))


def test_aggregate_means_over_sims_with_std_and_windows():
    torch.manual_seed(0)
    x, y = torch.rand(10, 2, 16, 16), torch.rand(10, 2, 16, 16)  # SSIM needs 11 x 11
    aggregate = Aggregate(["a", "b"], "cpu")
    v = torch.stack([aggregate.add(x, x)[0].mean(), aggregate.add(y, x)[0].mean()])
    m = aggregate.metrics(stride=2)
    assert m["vrmse_a"] == 0.5 * field_metrics(y, x)[0, 0] and abs(m["vrmse_std"] - v.std(unbiased=False)) < 1e-6
    assert len(m["vrmse_per_frame"]) == 10 and "vrmse_frames_2_8" in m and "vrmse_frames_27_56" not in m
    given = Aggregate(["a", "b"], "cpu", given=5)
    given.add(torch.cat([y[:5], x[5:]]), x)  # the first five frames shown: exact there, not scored
    assert given.metrics()["vrmse_a"] == 0 and len(given.metrics()["vrmse_per_frame"]) == 10
    clips = Aggregate(["a", "b"], "cpu", trajectory=False)
    clips.add(y, x)
    assert "vrmse_per_frame" not in clips.metrics() and abs(clips.metrics()["vrmse_b"] - 2 * m["vrmse_b"]) < 1e-6


def test_benchmark_scores_the_cfd_bench_frames_the_stride_reaches(tmp_path):
    from windinet.wan.dit.benchmark import SHARED, Benchmark

    stats = {f: {"mean": 0.0, "std": 1.0} for f in EULERMQ.fields}
    (tmp_path / "stats.json").write_text(json.dumps({"data_normalization_stats": stats}))
    x = torch.rand(49, 4, 16, 16)  # 101 frames at stride 2: frames 0, 2, ..., 96
    bench = Benchmark(tmp_path / "stats.json", list(EULERMQ.fields), 2, "cpu")
    bench.add(x, x)
    m = bench.metrics()
    assert m["bench_frames"] == [10, 20, 30, 40, 50, 60, 70, 80, 90]
    assert m["bench_rmse"] < 1e-3 and m["bench_vrmse"] < 1e-3 and m["bench_ssim"] < 1e-3 and m["bench_mlw"] < 1e-3
    bench.add(x, torch.zeros_like(x))
    m = bench.metrics()
    assert m["bench_vrmse"] > 0.4 and abs(m["bench_vrmse_std"] - m["bench_vrmse"]) < 1e-5  # 0 and ~1: std = mean
    vae = Benchmark(tmp_path / "stats.json", list(EULERMQ.fields), 2, "cpu", prefix="bench_vae")
    assert "bench_vae_frames" in vae.metrics()
    shared = Benchmark(tmp_path / "stats.json", list(EULERMQ.fields), 4, "cpu", prefix="bench_shared", frames=SHARED)
    assert shared.scored == [5, 10, 15, 20]


def test_latent_windows_and_start_clips(tmp_path):
    (tmp_path / "train").mkdir()
    torch.save({"latents": torch.arange(13.0).view(1, 1, 13, 1, 1), "gamma": 1.3}, tmp_path / "train" / "a_gamma1.3.pt")
    (tmp_path / "encode.yaml").write_text("config: {data: {}}\nstarts: [0, 4, 8]\n")
    assert with_augmentations(tmp_path, ["a_gamma1.3"]) == ["a_gamma1.3", "a_gamma1.3_t4", "a_gamma1.3_t8"]
    assert latent_id("a_gamma1.3", 2, 4) == "a_gamma1.3_s2_t4"
    assert latent_id("a_gamma1.3") == "a_gamma1.3" and latent_id("a_gamma1.3", 5) == "a_gamma1.3_s5"
    z, _ = LatentDataset(tmp_path, "train", ["a_gamma1.3"], window=7)[0]
    assert z.shape[2] == 7 and torch.equal(z.flatten(), torch.arange(z[0, 0, 0, 0, 0], z[0, 0, 0, 0, 0] + 7))


def test_rollout_chunks_regrow_the_history():
    from types import SimpleNamespace as NS

    class Cond:
        def __call__(self, g):
            return g

        uncond = __call__

    calls = []
    sample = lambda c, u, shape, *a, **k: calls.append(k["context"]) or torch.full(shape, float(len(calls)))  # noqa: E731
    dit = NS(device=torch.device("cpu"), sample=sample, delta_scale=None)
    for chunk, k, contexts in ((4, 1, [1, 5, 9]), (6, 1, [1, 7]), (4, 2, [2, 6, 10]), (None, 1, [1])):
        calls.clear()
        cfg = NS(cond=NS(context=k), sampling=NS(steps=1, guidance=1.0, chunk=chunk, history_noise=0.0, stochastic=0.0))
        cfg.model, cfg.train = NS(precision="fp32"), NS(seed=0)
        out = rollout(dit, Cond(), torch.zeros(2, 1, 16, 13, 4, 4), torch.ones(2), cfg, [0, 1])
        assert out.shape == (2, 1, 16, 13, 4, 4) and calls == contexts
        frames = out[0, 0, 0, :, 0, 0].tolist()  # which generation each frame comes from
        assert frames == [min(max(1, (f - k) // (chunk or 13) + 1), len(contexts)) for f in range(13)]


def test_frame_sigmas_plain_and_noisy_history():
    draws = torch.rand(4, 13) * 0.8 + 0.1
    sigma, target = frame_sigmas(draws, 1, None)
    assert torch.equal(sigma[:, 0], torch.zeros(4)) and torch.equal(sigma[:, 1:], draws[:, :1].expand(4, 12))
    assert torch.equal(target, torch.arange(13)[None].expand(4, 13) >= 1)
    sigma, target = frame_sigmas(draws, 1, 0.1)
    assert torch.equal(sigma[:, 0], torch.zeros(4)) and torch.equal(sigma[target], draws[target])
    history = ~target & (torch.arange(13) > 0)
    assert (sigma[history] <= 0.1).all() and target[:, -1].all() and not target[:, 0].any()
    assert all(len(set(sigma[i][history[i]].tolist())) <= 1 for i in range(4))  # one history level per sample
    assert torch.equal(frame_sigmas(draws, 3, None)[0][:, :3], torch.zeros(4, 3))
    x0 = torch.zeros(4, 2, 13, 3, 3)
    assert torch.equal(noised(x0, torch.ones_like(x0), sigma)[:, 0, :, 0, 0], sigma)


def test_clean_estimate_recovers_z_and_gate_picks_the_lowest_sigmas():
    z, noise = torch.randn(3, 2, 4, 2, 2), torch.randn(3, 2, 4, 2, 2)
    sigma = torch.tensor([[0.0, 0.9, 0.9, 0.9], [0.0, 0.3, 0.3, 0.3], [0.0, 0.6, 0.6, 0.6]])
    assert torch.allclose(clean_estimate(noised(z, noise, sigma), velocity_target(z, noise), sigma), z, atol=1e-6)
    assert gated(sigma, 0.8, 1).tolist() == [1] and gated(sigma, 0.8, 3).tolist() == [1, 2]
    assert len(gated(sigma, 0.2, 2)) == 0


def test_l1_gdl_ignores_an_offset_and_charges_a_displaced_edge():
    x = torch.zeros(1, 1, 8, 8)
    x[..., 4:, :] += 1  # an edge in each direction (a constant direction would have no difference std)
    x[..., :, 4:] += 1
    assert torch.isclose(l1_gdl(x + 0.1, x), torch.tensor(0.1))
    shifted = x.roll(1, dims=-1)
    assert l1_gdl(shifted, x) > (shifted - x).abs().mean()


def test_amse_does_not_reward_damping_an_unknown_scale():
    torch.manual_seed(0)
    x, guess = torch.randn(2, 4, 2, 32, 32)  # the guess has the right power but no coherence with x
    mse = (guess - x).square().mean()
    assert amse(x, x) < 1e-6 and amse(guess, x) >= mse - 1e-4
    assert (0.5 * guess - x).square().mean() < mse  # MSE prefers the damped guess
    assert amse(0.5 * guess, x) > amse(guess, x)  # AMSE does not
    assert torch.isclose(amse(1.5 * x, x), (0.5 * x).square().mean(), rtol=1e-4)  # coherent: amplitude error only


def test_with_latents_pairs_each_sims_frames_with_its_rollout(tmp_path):
    import h5py

    from windinet.data.frames import FrameSampler
    from windinet.wan.dit.latents import WithLatents

    ids = ["0000_gamma1.3", "0001_gamma1.4"]
    with h5py.File(tmp_path / "t.h5", "w") as f:
        for j, sid in enumerate(ids):
            for name in EULERMQ.fields:
                f.create_dataset(f"{sid}/{name}", data=torch.full((11, 1, 2, 2), float(j)))
    (tmp_path / "r" / "test").mkdir(parents=True)
    for j, sid in enumerate(ids):
        torch.save(
            {"latents": torch.full((2, 16, 3, 1, 1), float(j)), "gamma": 1.3}, tmp_path / "r" / "test" / f"{sid}.pt"
        )
    data = WithLatents(
        FrameSampler(tmp_path / "t.h5", ids[::-1], None, 0, stride=2, spec=EULERMQ), tmp_path / "r", "test"
    )
    frames, z = data[0]
    assert frames.shape[0] == 5 and z.shape == (2, 16, 3, 1, 1)
    assert frames.flatten()[0] == z.flatten()[0] == 1.0  # the same sim: the second id comes first


def test_front_distance_and_shift_see_a_moved_front():
    from windinet.experiment.metrics import shift
    from windinet.wan.vae.loss import front_distance, window

    x = torch.zeros(2, 1, 64, 64)
    x[..., 30:] = 1.0
    moved = torch.zeros_like(x)
    moved[..., 33:] = 1.0
    assert front_distance(x, x) < 0.01 < front_distance(moved, x)
    assert shift(x, x, tile=16).item() == 0 and shift(moved, x, tile=16).item() == 3
    starts = {window(torch.zeros(1, 1, 16, 13, 2, 2), 6).start for _ in range(200)}
    assert starts <= set(range(1, 8)) and len(starts) > 3


def test_only_the_corrector_refuses_srba_oooo(monkeypatch, tmp_path):
    root, overrides = Path(__file__).parents[1], ["data.dataset=srba_oooo", f"results_dir={tmp_path}"]
    modules = {}
    for script in ("train_dit", "train_corrector"):
        spec = importlib.util.spec_from_file_location(script, root / f"scripts/wan/{script}.py")
        modules[script] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(modules[script])
    with pytest.raises(typer.BadParameter, match="srba_oooo"):  # before start_run: no folder blocks the name
        modules["train_corrector"].main(root / "configs/wan/corrector.yaml", "x", "x", overrides, False)
    assert not (tmp_path / "x").exists()

    def started(*args):
        raise RuntimeError("started")

    monkeypatch.setattr(modules["train_dit"], "start_run", started)
    with pytest.raises(RuntimeError, match="started"):
        modules["train_dit"].main(root / "configs/wan/dit.yaml", "x", "x", overrides, False)


def stand_in_evaluation(monkeypatch, module, spec, ids, truth, tmp_path, labels):
    """`module.evaluate` of sims `ids` with fields `truth` [B, F, C, H, W], without Wan: the latents are the fields
    as [B, 1, C, F, H, W], decode undoes that and the DiT returns them plus seeded noise; two samples, cfd_bench on."""
    from types import SimpleNamespace as NS

    from torch.utils.data import DataLoader, TensorDataset

    latents = truth.transpose(1, 2)[:, None]

    def sample(c, u, shape, steps, guidance, seeds, **kwargs):
        noise = torch.stack([torch.randn(shape[1:], generator=torch.Generator().manual_seed(s)) for s in seeds])
        return latents.flatten(1, 2) + 0.1 * noise  # all sims in one batch

    def cond(c):
        return c

    cond.uncond = cond
    monkeypatch.setattr(module, "decode", lambda adapter, z, vaes: z[:, 0].transpose(1, 2))
    stats = {f: {"mean": 0.0, "std": 1.0} for f in spec.fields}
    (tmp_path / "stats.json").write_text(json.dumps({"data_normalization_stats": stats}))
    sampling = NS(samples=2, seed=0, steps=1, guidance=1.0, chunk=None, history_noise=0.0, stochastic=0.0)
    data = NS(spec=spec, frame_stride=1, stats_json=tmp_path / "stats.json")
    cfg = NS(cond=NS(context=1), data=data, sampling=sampling, score_from=2, benchmark=True, model=NS(precision="fp32"))
    conditions = TensorDataset(latents, torch.tensor([list(spec.scalars(i).values()) for i in ids]))
    conditions.ids = ids
    loaders = DataLoader(conditions, len(ids)), DataLoader(TensorDataset(truth, truth), len(ids))
    dit = NS(device=torch.device("cpu"), sample=sample, delta_scale=None)
    adapter = NS(fields=list(spec.fields), select=lambda x: x)
    return module.evaluate(dit, cond, adapter, [], loaders, labels, cfg)


def test_dit_evaluation_of_a_uniform_start_per_scalar(monkeypatch, tmp_path):
    cells = ((1.3, 1), (1.3, 2), (1.7, 1), (2.1, 3))
    ids = [f"{i:03d}_Mas{m:.2f}_sb{b}_Ax0.0129_Ay0.0081_Ar0.0035_BxNaN_ByNaN_BrNaN" for i, (m, b) in enumerate(cells)]
    truth = torch.randn(len(ids), 21, 4, 16, 16)
    truth[:, :6, SRBA_OOOO.fields.index("velocityY")] = 0  # before the shock: a frame's own variance would be ~0
    metrics, examples = stand_in_evaluation(monkeypatch, evaluate, SRBA_OOOO, ids, truth, tmp_path, (1.3, 2.1))
    curve = torch.tensor(metrics["vrmse_per_frame"])
    assert curve.isfinite().all() and curve.max() < 1 and set(examples) == {1.3, 2.1}
    for name, counts in (("mach", {1.3: 2, 1.7: 1, 2.1: 1}), ("bubbles", {1: 2, 2: 1, 3: 1})):
        means = sum(n * metrics[f"vrmse_mean_{name}{v:g}"] for v, n in counts.items()) / len(ids)
        assert means == pytest.approx(metrics["vrmse_mean"], rel=1e-6)
