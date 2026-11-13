"""Speed options of full finetuning (--tf32, --compile, --checkpoint-layers, --latents): same result, how fast,
how much memory. One GPU, the real 4.5B DiT, one training-shaped sample (100 frames: 30 latents, 5 chunks), ~15 min.

    sbatch jobs/jupiter/magi1/quick_test.sbatch -m magi_finetuning.tests.test_speed

Latent cache: the stored latents of a few train sims must equal what training encodes on the fly
(data.batch_to_latents), and the time of that encoding is what --latents saves per sample.

DiT: every variant runs forward + backward of velocity_loss on the same input and noise, exactly as
train_full.training_micro_step (bf16 autocast, fp32 master weights, gradients kept allocated). The 8-bit AdamW
state is allocated first (one lr = 0 step), so peak memory is the training peak up to DDP's small extras.

Correctness: the loss and the gradient of every variant are compared with the eager reference: gradient as
||g - g_ref|| / ||g_ref|| over every 16th entry of every parameter, next to the noise floor (the eager
reference run twice). bf16 rounds after every op, so equivalent implementations differ by a little; a wrong
graph (e.g. one layer's weights reused for all) differs by ~100 %. Compile must also really compile: a few
graphs reused by all 34 layers, no graph breaks.
"""

import json
import time
from pathlib import Path

import torch

import magi_finetuning.compat  # noqa: F401  (stand-ins before MAGI imports)
from magi_finetuning.data import (CachedLatents, Normalization, batch_to_latents, cached_batch_to_latents, indices_of,
                                  load_split, stratified_subset)
from magi_finetuning.dit import (Model, add_noise, compile_layers, enable_gradient_checkpointing, gamma_embedding,
                                 keep_image_clean, load_dit, set_fp32_matmul, velocity_loss)
from magi_finetuning.settings import CHUNK_WIDTH, SCRATCH, TRAIN_DATA
from magi_finetuning.train.train_full import make_adamw8bit
from magi_finetuning.utils.runtime import init_magi_parallel_state
from magi_finetuning.vae import load_vae_encoder
from windinet.training.shockwave_data import CHANNEL_NAMES, ShockWaveDataset

RESULTS = []
NUM_CHUNKS, NUM_BODY, TIMED = 5, 25, 3
SCALE = 0.18215  # MAGI-1 runtime_config.scale_factor


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)


COMPILED = {"on": False}


def configure(dit, tf32: bool, compiled: bool, checkpoint_layers: int | None) -> None:
    """Set the three options on the same model. compile_layers / enable_gradient_checkpointing patch instance
    attributes; the compiled functions are kept while only the checkpointing changes (no recompilation)."""
    for layer in dit.videodit_blocks.layers:
        layer.__dict__.pop("forward", None)  # the checkpoint wrapper
        if COMPILED["on"] and not compiled:
            layer.__dict__.pop("attn_post_process", None)
            for name in ("get_q", "get_kv", "get_xqkv"):
                layer.self_attention.__dict__.pop(name, None)
    set_fp32_matmul(tf32)
    if compiled and not COMPILED["on"]:
        compile_layers(dit)
    COMPILED["on"] = compiled
    enable_gradient_checkpointing(dit, checkpoint_layers)


def make_input():
    g = torch.Generator(device="cuda").manual_seed(0)
    x1 = torch.randn(1, 16, NUM_CHUNKS * CHUNK_WIDTH, 32, 32, device="cuda", generator=g)
    noise = torch.randn(x1.shape, device="cuda", generator=g)
    t = torch.tensor([[0.8, 0.6, 0.4, 0.2, 0.05]], device="cuda")
    xt = keep_image_clean(add_noise(x1, t, noise), x1)
    gamma = torch.tensor([[1.5]], device="cuda")
    return xt, t, gamma, x1, noise


def forward_backward(model, inputs) -> float:
    xt, t, gamma, x1, noise = inputs
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()  # stays allocated during forward, as in training (zero_grad(set_to_none=False))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        v = model(xt, t, gamma, NUM_CHUNKS, CHUNK_WIDTH * 16 * 16)
    loss = velocity_loss(v, x1, noise, NUM_BODY)
    loss.backward()
    return loss.item()


def grads(model) -> torch.Tensor:
    """Every 16th entry of every parameter's gradient (~280 M values: the full copy would not fit next to training)."""
    return torch.cat([p.grad.flatten()[::16] for p in model.parameters()])


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a - b).norm() / b.norm()).item()


def test_latent_cache(folder: Path, sims: int = 4) -> None:
    """Cached latents == the latents training encodes, for a few train sims spread over gamma; encode time."""
    stats = json.loads((folder / "stats.json").read_text())
    encoder = load_vae_encoder(stats["vae_ckpt"])
    normalization = Normalization.of_vae_checkpoint(stats["vae_ckpt"])
    dataset = ShockWaveDataset(TRAIN_DATA, num_sim_frames=101)
    train_ids = stratified_subset(load_split()["train"], None)
    picked = train_ids[::len(train_ids) // sims][:sims]
    cached = CachedLatents(folder, "train", picked, 100, stats["vae_ckpt"])
    worst, same_rest, seconds = 0.0, True, []
    for index, row in enumerate(indices_of(dataset, picked)):
        collate = torch.utils.data.default_collate
        sample = collate([dataset[row]])  # what the training loader yields, then moved to the GPU
        batch = {name: sample[name].cuda() for name in CHANNEL_NAMES}
        batch["meta"] = {"gamma": sample["meta"]["gamma"].cuda()}
        torch.cuda.synchronize()
        started = time.perf_counter()
        fresh = batch_to_latents(batch, encoder, normalization, SCALE)
        torch.cuda.synchronize()
        seconds.append(time.perf_counter() - started)
        stored = cached_batch_to_latents({k: v.cuda() for k, v in collate([cached[index]]).items()}, SCALE)
        worst = max(worst, (fresh[0] - stored[0]).abs().max().item())
        same_rest &= torch.equal(fresh[1], stored[1]) and fresh[2:] == stored[2:]
    check("--latents: cached latents == encoded on the fly", worst == 0.0, f"max abs diff {worst:.1e} over {sims} sims")
    check("--latents: same gamma, body latents, chunks, tokens", same_rest)
    print(f"  VAE encoding that --latents saves: {sum(seconds[1:]) / len(seconds[1:]):.3f} s per sample\n", flush=True)
    del encoder
    torch.cuda.empty_cache()


def run_variant(model, inputs, label: str, setup_bytes: int, want_grads: bool):
    """First call (incl. compile), then TIMED timed calls. Returns (loss, grads or None, s per sample, peak GB),
    or None on out of memory. Peak excludes the gradient copies this test holds for comparisons."""
    held = torch.cuda.memory_allocated() - setup_bytes
    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    try:
        loss = forward_backward(model, inputs)
        first = time.time() - start
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(TIMED):
            forward_backward(model, inputs)
        torch.cuda.synchronize()
    except torch.OutOfMemoryError:
        print(f"  {label}: out of memory", flush=True)
        torch.cuda.empty_cache()
        return None
    seconds = (time.time() - start) / TIMED
    peak = (torch.cuda.max_memory_allocated() - held) / 1e9
    print(f"  {label}: {seconds:.2f} s per sample (first call {first:.0f} s), peak {peak:.1f} GB, loss {loss:.6f}",
          flush=True)
    return loss, grads(model) if want_grads else None, seconds, peak


def close(a: float, b: float) -> bool:
    return abs(a - b) <= 0.01 * abs(b)


def main():
    torch.cuda.set_device(0)
    init_magi_parallel_state()
    test_latent_cache(SCRATCH / "magi1_latents_F3")
    dit, _ = load_dit()
    dit = dit.float().requires_grad_(True)
    model = Model(dit, gamma_embedding())
    optimizer = make_adamw8bit([p for p in model.parameters() if p.requires_grad], lr=0.0)
    inputs = make_input()
    forward_backward(model, inputs)  # gradients, so the optimizer allocates its state
    optimizer.step()                 # lr = 0, weight decay * lr = 0: weights unchanged
    setup_bytes = torch.cuda.memory_allocated()
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU memory {total:.0f} GB; weights + grads + 8-bit AdamW: {setup_bytes / 1e9:.1f} GB\n", flush=True)

    rows = {}  # label -> (s per sample, peak GB), or None (out of memory)

    def variant(label, tf32, compiled, ckpt, want_grads=True):
        configure(dit, tf32, compiled, ckpt)
        out = run_variant(model, inputs, label, setup_bytes, want_grads)
        rows[label] = None if out is None else out[2:]
        if out is None and want_grads:
            raise RuntimeError(f"{label}: out of memory in a correctness variant")
        return out

    ref = variant("today: fp32 proj, eager, recompute all 34", False, False, None)
    again = run_variant(model, inputs, "today, repeated (noise floor)", setup_bytes, True)
    floor = rel(again[1], ref[1])
    check("eager reference reproducible", floor < 2e-2, f"grad rel diff {floor:.2e} (noise floor)")
    limit = max(5 * floor, 5e-2)  # bf16 rounds after every op: equivalent implementations differ a little
    del again

    tf = variant("tf32, eager, recompute all 34", True, False, None)
    d = rel(tf[1], ref[1])
    check("tf32 == fp32 (loss within 1 %, gradient within 5 %)", close(tf[0], ref[0]) and d < limit,
          f"grad rel diff {d:.2e} (floor {floor:.2e}), loss {tf[0]:.6f} vs {ref[0]:.6f}")
    del ref

    half = variant("tf32, eager, recompute 17", True, False, 17)
    d = rel(half[1], tf[1])
    check("partial checkpointing: same loss and gradient", close(half[0], tf[0]) and d < limit,
          f"grad rel diff {d:.2e} (floor {floor:.2e})")
    del half

    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    comp = variant("tf32, compiled, recompute all 34", True, True, None)
    d = rel(comp[1], tf[1])
    check("compiled == eager (loss within 1 %, gradient within 5 %)", close(comp[0], tf[0]) and d < limit,
          f"grad rel diff {d:.2e} (floor {floor:.2e}), loss {comp[0]:.6f} vs {tf[0]:.6f}")
    counters = torch._dynamo.utils.counters
    graphs = counters["stats"].get("unique_graphs", 0)
    breaks = dict(counters["graph_break"])
    check("compile: few graphs shared by all 34 layers (not one per layer), no graph breaks",
          0 < graphs <= 12 and not breaks, f"unique graphs {graphs} (4 pieces x with / without grad = 8 expected), "
          f"graph breaks {breaks or 'none'}")
    print(f"  dynamo: stats {dict(counters['stats'])}, frames {dict(counters['frames'])}", flush=True)
    del tf, comp

    for ckpt in (17, 8, 0):
        variant(f"tf32, compiled, recompute {ckpt}", True, True, ckpt, want_grads=False)

    base = rows["today: fp32 proj, eager, recompute all 34"][0]
    print(f"\n| variant | s per sample | speedup | peak GB (of {total:.0f}) |\n|---|---|---|---|")
    for label, row in rows.items():
        print(f"| {label} | OOM | - | - |" if row is None else f"| {label} | {row[0]:.2f} | {base / row[0]:.2f}x | {row[1]:.1f} |")
    print("\nTraining = 8 samples per GPU per optimizer step (4 GPUs); DDP adds a little memory on top.")
    print(f"\n{sum(RESULTS)}/{len(RESULTS)} checks passed -> {'ALL PASS' if all(RESULTS) else 'FAILURES'}", flush=True)


if __name__ == "__main__":
    main()
