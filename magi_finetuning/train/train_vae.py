"""Stage 1: adapt MAGI-1's VAE to the 4 Euler fields before training the DiT on its latents.

Defaults reproduce Weihao Li's best LTX run (thesis) on MAGI-1:
    trained:  whole encoder + decoder (--train full), new pressure channel zero-initialized
    loss:     1.0 * RMSE + 50 * H1 + 0.01 * SDS   (no SSIM)
    SDS:      score distillation from the frozen stock MAGI-1 DiT (teacher) with a random, frozen
              gamma embedding. The teacher sees a noised copy of the latents (stop-gradient) and its
              predicted velocity is detached; the gradient reaches the encoder only through the
              target (z - noise), nudging the latents toward what MAGI-1's DiT can model. The clean
              first chunk is excluded, like the conditioning frame in the thesis.
    optim:    AdamW lr 5e-5, warmup 200 steps from 0.01x, cosine to 1e-6, clip 5.0, batch 32, seed 42,
              20 epochs over the train split; the checkpoint with the lowest val VRMSE is kept.
One sample = 96 consecutive frames of a trajectory (MAGI needs 4 chunks of 24), starting at a random frame
0..5 drawn anew every epoch, so all 101 frames are seen (frames 5..95 every time, 0 and 100 about 1 epoch in
6); validation uses frames 0..95. Encoded in 16-frame tiles exactly as the DiT path encodes.
--train io: only the two channel layers (patch_embed.proj, last_layer); everything else frozen.
--train adapter: the stock RGB VAE with windinet's 1x1 adapters around it (vae.load_vae_with_adapters);
    the encoder is frozen, so trained: input adapter (4 -> 3), whole decoder, output adapter (3 -> 4).

Normalization: --stats (yaml) and --log-fields (e.g. density pressure: normalize log(x)); stored in
every checkpoint's metadata, so the DiT and evaluation always use the VAE's own (data.Normalization).
--sds-weight 0 drops SDS (no teacher is loaded).

Options for the pressure problem (the official VAE learned to output "pressure ~ density" instead of
encoding pressure):
    --recon-loss vrmse_per_channel   train on the metric itself: per field, error / that field's variance in
                                     the sample, then the mean over fields (default rmse: one RMSE over all
                                     fields pooled, windinet's rmse_loss)
    --recon-loss l1                  mean absolute error over all fields (MAGI-1's own VAE loss, with KL/LPIPS/GAN)
    --h1-weight W                    weight of the H1 (gradient) term (Weihao: 50)
    --io-lr-mult M                   M x learning rate for the two 4-channel layers (encoder input, decoder output)
    --init random                    new pressure filters random with the RGB filters' spread (not zero, not mean)
    --decoder-extra-channels K       rebuild each output pixel from 4 + K values instead of MAGI's 4 (vae.WideDecoder)
    --decoder-refine-channels C      plus a residual 3-layer conv stack of width C on those values (smooths patch seams)
Every validation also prints how big the encoder's pressure input filter is relative to the RGB ones, and how
similar the decoder's pressure output filter is to its density one (pressure_path).

Long runs: --max-minutes stops cleanly and writes <ckpt-dir>/resume.pt; rerun with --resume to go on,
with the same data order and random numbers as one long job (utils/resume.py).

Run from the repo root, env active:
    accelerate launch --multi_gpu --num_processes 4 -m magi_finetuning.train.train_vae --ckpt-dir $SCRATCH/...
Then the DiT:  ... -m magi_finetuning.train.finetune --full-finetune --vae-ckpt <ckpt-dir>/vae_best.safetensors

Read `run()` first; every helper below it does one thing.
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from accelerate.utils import set_seed
from safetensors.torch import save_file
from torch.utils.data import DataLoader, Dataset

from magi_finetuning.data import LOGGABLE, Normalization, indices_of, load_split, stratified_subset
from magi_finetuning.dit import Model, add_noise, gamma_embedding, load_dit, sample_chunk_t
from magi_finetuning.settings import CHUNK_WIDTH, NORMALIZATION_STATS, OUTPUTS, TRAIN_DATA
from magi_finetuning.utils.resume import EpochShuffle, batches_from, restore_rng_states, save_rng_states
from magi_finetuning.utils.runtime import JobTimer, check_ckpt_dir, init_distributed, shutdown
from magi_finetuning.vae import build_vae, enable_vae_gradient_checkpointing, reconstruct
from windinet.losses import SSIMLoss, h1_seminorm_loss, rmse_loss, vrms_per_channel
from windinet.training.shockwave_data import CHANNEL_NAMES, ShockWaveDataset, parse_gamma

LOSS_WEIGHTS = {"rmse": 1.0, "vrmse": 1.0, "l1": 1.0, "h1": 50.0, "ssim": 0.0, "sds": 0.0}  # sds: --sds-weight (Weihao: 0.01);
# only one of rmse / vrmse / l1 is computed (--recon-loss); h1: --h1-weight
EFFECTIVE_BATCH = 32
SEED = 42
MAX_GRAD_NORM = 5.0
IO_LAYERS = ("encoder.patch_embed.proj", "decoder.last_layer", "decoder.extra_proj", "decoder.refine")  # 4-channel layers + new head


def parse_args():
    p = argparse.ArgumentParser(description="Adapt MAGI-1's VAE to the 4 Euler fields.")
    p.add_argument("--data", default=TRAIN_DATA)
    p.add_argument("--out", default=str(OUTPUTS / "vae"), help="args and small files (on SCRATCH)")
    p.add_argument("--ckpt-dir", default=None,
                   help="checkpoints + resume state; required for --train full, never under home/IG_work")
    p.add_argument("--train", choices=["full", "io", "adapter"], default="full",
                   help="full: whole encoder + decoder (Weihao); io: only the two channel layers; "
                        "adapter: frozen encoder, 4->3 / 3->4 adapters + decoder")
    p.add_argument("--init", choices=["zeros", "mean", "random"], default="zeros",
                   help="how the new pressure channel starts (full/io; adapters always start as identity)")
    p.add_argument("--stats", default=str(NORMALIZATION_STATS), help="normalization statistics (yaml)")
    p.add_argument("--log-fields", nargs="*", choices=LOGGABLE, default=[],
                   help="normalize log(x) for these fields (needs log_<field> stats in --stats)")
    p.add_argument("--sds-weight", type=float, default=LOSS_WEIGHTS["sds"], help="0: no SDS, no teacher")
    p.add_argument("--recon-loss", choices=["rmse", "vrmse_per_channel", "l1"], default="rmse",
                   help="rmse: pooled over all fields (Weihao); vrmse_per_channel: per field / its variance, then "
                        "mean; l1: mean absolute error (MAGI-1's VAE loss)")
    p.add_argument("--h1-weight", type=float, default=LOSS_WEIGHTS["h1"], help="weight of the H1 gradient term")
    p.add_argument("--decoder-extra-channels", type=int, default=0,
                   help="K > 0: each output pixel from 4 + K decoder values instead of MAGI's 4 (full/io only)")
    p.add_argument("--decoder-refine-channels", type=int, default=0,
                   help="C > 0: residual conv stack (width C) after the decoder's unpatchify (full/io only)")
    p.add_argument("--io-lr-mult", type=float, default=1.0,
                   help="learning-rate multiplier for the two 4-channel layers (encoder input, decoder output)")
    p.add_argument("--clip-frames", type=int, default=96, help="frames per sample; multiple of 48 (VAE tiles + DiT chunks)")
    p.add_argument("--epochs", type=float, default=20, help="passes over the train split (sets the step count)")
    p.add_argument("--steps", type=int, default=None, help="override --epochs")
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--shift", type=float, default=3.0, help="SDS teacher time sampling, same as DiT training")
    p.add_argument("--max-samples", type=int, default=None, help="train on ~N sims, spread evenly over all gammas")
    p.add_argument("--eval-samples", type=int, default=90, help="val sims for val VRMSE (spread over gammas)")
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--max-minutes", type=float, default=None, help="stop cleanly (and save resume state) after this")
    p.add_argument("--resume", action="store_true",
                   help="continue from <ckpt-dir>/resume.pt if it exists (otherwise start fresh)")
    args = p.parse_args()
    if args.train == "adapter" and (args.decoder_extra_channels or args.decoder_refine_channels):
        raise ValueError("--decoder-extra-channels widens the grown VAE's decoder; not with --train adapter")
    if args.train == "adapter" and args.init != "zeros":
        raise ValueError("--train adapter starts as identity (pressure ignored / predicted 0); --init mean is for full/io")
    if args.clip_frames % 48:
        raise ValueError("--clip-frames must be a multiple of 48 (16-frame VAE tiles and 24-frame DiT chunks)")
    return args


def run(args):
    timer = JobTimer()  # setup / training time -> <ckpt-dir>/timing.json
    # 1. Multi-GPU setup: effective batch 32 = 1 trajectory per GPU x grad_accum x number of GPUs
    num_gpus = int(os.environ.get("WORLD_SIZE", "1"))
    accelerator = init_distributed(EFFECTIVE_BATCH // num_gpus)
    set_seed(SEED, device_specific=True)
    loss_weights = {**LOSS_WEIGHTS, "sds": args.sds_weight, "h1": args.h1_weight}

    # 2. Data (fixed gamma-stratified train/val split), normalization, the step count from --epochs
    train_set, eval_set, normalization = make_datasets(args)
    accelerator.print(f"normalization: {normalization.describe()}")
    if args.steps is None:
        args.steps = math.ceil(args.epochs * len(train_set) / EFFECTIVE_BATCH)
    output_dir = prepare_output_dir(args, accelerator, len(train_set), loss_weights)
    sampler = EpochShuffle(len(train_set), SEED)  # exactly rebuildable order (utils/resume.py)
    loader = DataLoader(train_set, batch_size=1, sampler=sampler, num_workers=4, drop_last=True)

    # 3. Models: the trainable 4-channel VAE, and the frozen SDS teacher (stock MAGI-1 DiT)
    magi_vae = build_vae(vae_mode(args), args.init, args.decoder_extra_channels, args.decoder_refine_channels)
    trainable_params = choose_trainable(magi_vae, args.train)
    enable_vae_gradient_checkpointing(magi_vae)
    vae = Autoencoder(magi_vae).cuda()
    teacher, magi_config = load_sds_teacher() if loss_weights["sds"] > 0 else (None, None)
    num_params = sum(p.numel() for p in trainable_params)
    accelerator.print(f"trainable params: {num_params:,} ({args.train}), {args.steps} steps, "
                      f"{len(train_set)} train / {len(eval_set[1])} val sims")
    checkpoint_dir = prepare_checkpoint_dir(args, accelerator, output_dir, num_params)
    ssim_loss = SSIMLoss(channels=len(CHANNEL_NAMES)).cuda() if loss_weights["ssim"] > 0 else None
    saving = (checkpoint_dir, args.train, normalization, vae_mode(args),
              (args.decoder_extra_channels, args.decoder_refine_channels))  # where and how checkpoints are written

    # 4. Optimizer and learning-rate schedule (linear warmup, then cosine to min_lr)
    optimizer = torch.optim.AdamW(parameter_groups(magi_vae, args), lr=args.lr, weight_decay=0.0)
    scheduler = warmup_then_cosine(optimizer, args)
    state = {"step": 0, "best_val": float("inf"), "best_step": None, "micro_batches_done": 0, "world_size": None}
    if args.resume and (Path(checkpoint_dir) / "resume.pt").is_file():
        state = load_resume_state(checkpoint_dir, magi_vae, optimizer, scheduler, EFFECTIVE_BATCH // num_gpus)
        if state["step"] >= args.steps:  # a spare job in the chain after the run already finished
            accelerator.print(f"run already finished at step {state['step']}; exiting")
            shutdown(accelerator)
            return
    # The scheduler is deliberately NOT passed to prepare(): Accelerate would step it once per GPU.
    vae, optimizer, loader = accelerator.prepare(vae, optimizer, loader)
    # Same data order (and crop starts) as one long job: rebuild the epoch, skip what this GPU already saw.
    batches = batches_from(loader, accelerator, state["micro_batches_done"], epoch_aware=[sampler, train_set])
    if state["step"] > 0:
        restore_random_state(accelerator, checkpoint_dir, state)

    # 5. Training loop
    recent_losses = []
    steps_from = state["step"]
    timer.training_started()
    start_time = last_step_time = time.time()
    for clip, gamma in batches:
        state["micro_batches_done"] += 1
        target = normalization.apply(clip.float())
        losses, grad_norm = training_micro_step(
            accelerator, vae, optimizer, trainable_params, target, gamma.float()[:, None],
            ssim_loss, teacher, magi_config, loss_weights, args,
        )
        recent_losses.append(losses)

        # Only every grad_accum-th micro-step is a real optimizer step.
        if not accelerator.sync_gradients:
            continue
        scheduler.step()
        state["step"] += 1
        step = state["step"]

        now = time.time()
        if step == 1 or step % args.log_every == 0:
            log_step(accelerator, step, recent_losses, grad_norm, scheduler, now - last_step_time)
        recent_losses = []
        last_step_time = now

        if step == args.steps or step % args.eval_every == 0:
            val_vrmse = evaluate(accelerator, vae, eval_set, normalization, step, output_dir / "val_history.jsonl")
            keep_if_best(accelerator, vae, saving, step, val_vrmse, state)
        if step >= args.steps:
            break
        if args.max_minutes is not None and all_gpus_out_of_time(start_time, args.max_minutes):
            accelerator.print(f"--max-minutes {args.max_minutes} reached at step {step}; saving resume state")
            break

    save_weights(accelerator, vae, checkpoint_dir / "vae_last.safetensors", saving)
    if accelerator.is_main_process and state["step"] >= args.steps:  # read by data.check_vae_finished
        (checkpoint_dir / "finished.json").write_text(json.dumps(
            {"steps": state["step"], "best_step": state["best_step"], "best_val_vrmse": state["best_val"]}))
    save_resume_state(accelerator, vae, optimizer, scheduler, state, checkpoint_dir)
    if accelerator.is_main_process:
        timer.record(checkpoint_dir / "timing.json", "vae", steps_from, state["step"], num_gpus)
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    accelerator.print(f"done: step {state['step']}/{args.steps}, best val VRMSE {state['best_val']:.4f} "
                      f"at step {state['best_step']}, peak GPU memory {peak_gb:.1f} GB (rank 0)")
    shutdown(accelerator)


# ---------------------------------------------------------------- setup

def prepare_output_dir(args, accelerator, num_train: int, loss_weights: dict) -> Path:
    """Records the run's arguments, loss weights and data sizes in <out>/args.yaml."""
    output_dir = Path(args.out)
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        record = {**vars(args), "loss_weights": loss_weights, "effective_batch": EFFECTIVE_BATCH,
                  "num_train_sims": num_train}
        (output_dir / "args.yaml").write_text(yaml.safe_dump(record))
    return output_dir


def choose_trainable(vae, mode: str):
    """full: everything; io: only the two channel layers; adapter: input adapter + decoder (with its
    output adapter), the RGB encoder frozen. Everything else is frozen."""
    vae.requires_grad_(mode == "full")
    if mode == "io":
        vae.encoder.patch_embed.proj.requires_grad_(True)
        vae.decoder.last_layer.requires_grad_(True)
    if mode == "adapter":
        vae.encoder.in_adapter.requires_grad_(True)
        vae.decoder.requires_grad_(True)
    vae.eval()  # the VAE has no dropout/batch-norm, so this only makes the mode explicit
    return [p for p in vae.parameters() if p.requires_grad]


def vae_mode(args) -> str:
    """Architecture written into the checkpoints: "adapter", "wide" (--decoder-extra-channels > 0) or "grown"."""
    if args.train == "adapter":
        return "adapter"
    return "wide" if args.decoder_extra_channels > 0 or args.decoder_refine_channels > 0 else "grown"


def parameter_groups(vae, args) -> list[dict]:
    """The two 4-channel layers (IO_LAYERS) get lr x --io-lr-mult, everything else lr. The scheduler scales
    both groups by the same warmup/cosine factor, so the ratio stays fixed."""
    io = [p for n, p in vae.named_parameters() if p.requires_grad and n.startswith(IO_LAYERS)]
    rest = [p for n, p in vae.named_parameters() if p.requires_grad and not n.startswith(IO_LAYERS)]
    groups = [{"params": rest, "lr": args.lr}]
    if io:
        groups.append({"params": io, "lr": args.lr * args.io_lr_mult})
    return groups


def load_sds_teacher():
    """Stock MAGI-1 DiT (released weights, no finetuning) + a random gamma embedding, all frozen.
    Seeded identically on every GPU so all ranks use the same teacher."""
    dit, magi_config = load_dit()
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(42)
        teacher = Model(dit, gamma_embedding())
    teacher.cuda().eval().requires_grad_(False)
    return teacher, magi_config


def prepare_checkpoint_dir(args, accelerator, output_dir: Path, num_params: int) -> Path:
    """io checkpoints (~4 MB) may live in <out>/checkpoints; full and adapter ones (bf16 weights, plus
    a resume state several GB large) need --ckpt-dir outside home/IG_work, checked for free space first."""
    if args.ckpt_dir is None:
        if args.train != "io":
            raise ValueError(f"--train {args.train} needs --ckpt-dir (outside home/IG_work)")
        path = output_dir / "checkpoints"
        path.mkdir(parents=True, exist_ok=True)
        return path
    if not accelerator.is_main_process:
        return Path(args.ckpt_dir)
    args.save_optimizer = True  # check_ckpt_dir's estimate: weights + optimizer state
    return check_ckpt_dir(args, num_params, n_saves=3)  # best + last + resume


class Autoencoder(nn.Module):
    """forward(video) = (reconstruction, latent). A plain forward is what DDP needs to sync gradients."""

    def __init__(self, magi_vae):
        super().__init__()
        self.magi_vae = magi_vae

    def forward(self, video):
        return reconstruct(self.magi_vae, video)


class Trajectories(Dataset):
    """One simulation per item: (fields (4, clip_frames, H, W) raw, gamma). With random_start the clip
    starts at a random frame in [0, 101 - clip_frames], so the few frames past 96 are seen too. That
    start depends only on (epoch, item), not on the data-loader workers, so a resumed job crops exactly
    as an uninterrupted one would (start_epoch is called by resume.batches_from)."""

    def __init__(self, base: ShockWaveDataset, indices, clip_frames: int, random_start: bool):
        self.base, self.indices = base, list(indices)
        self.clip_frames, self.random_start = clip_frames, random_start
        self.epoch = 0

    def start_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        sample = self.base[self.indices[i]]
        fields = torch.stack([sample[name] for name in CHANNEL_NAMES], dim=0)
        last_start = fields.shape[1] - self.clip_frames
        generator = torch.Generator().manual_seed(SEED * 1_000_003 + self.epoch * 100_003 + i)
        start = int(torch.randint(0, last_start + 1, (), generator=generator)) if self.random_start else 0
        return fields[:, start:start + self.clip_frames], torch.tensor(parse_gamma(sample["id"]))


def make_datasets(args):
    """Train on the fixed train split, evaluate on the fixed val split (magi_finetuning/data.py, gamma-
    stratified 90/10). --max-samples / --eval-samples take an even spread over all gammas.
    Returns (train set, (eval set, eval indices), normalization)."""
    normalization = Normalization(str(Path(args.stats).resolve()), tuple(args.log_fields))  # stored in checkpoints
    normalization.mean_std  # fail now if the yaml lacks a needed entry (e.g. log_pressure)
    base = ShockWaveDataset(args.data)
    split = load_split()
    train_ids = stratified_subset(split["train"], args.max_samples)
    eval_ids = stratified_subset(split["val"], args.eval_samples)
    train_set = Trajectories(base, indices_of(base, train_ids), args.clip_frames, random_start=True)
    eval_set = Trajectories(base, indices_of(base, eval_ids), args.clip_frames, random_start=False)
    return train_set, (eval_set, list(range(len(eval_ids)))), normalization


def warmup_then_cosine(optimizer, args):
    """Linear warmup from 1% of lr over --warmup steps, then cosine down to --min-lr at the last step."""
    def factor(step):
        if step < args.warmup:
            return 0.01 + 0.99 * step / max(args.warmup, 1)
        progress = min(1.0, (step - args.warmup) / max(args.steps - args.warmup, 1))
        floor = args.min_lr / args.lr
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


# ---------------------------------------------------------------- one training step

def reconstruction_losses(reconstruction, target, ssim_loss, recon_loss: str) -> dict:
    """RMSE (pooled over all fields) or per-field VRMSE, plus H1 (and SSIM if weighted), in fp32.
    H1 is the mean over all fields of the squared gradient error, the same pooled or per field."""
    reconstruction = reconstruction.float()
    if recon_loss == "vrmse_per_channel":
        losses = {"vrmse": vrms_per_channel(reconstruction, target).mean()}  # = the validation metric
    elif recon_loss == "l1":
        losses = {"l1": (reconstruction - target).abs().mean()}
    else:
        losses = {"rmse": rmse_loss(reconstruction, target)}
    losses["h1"] = h1_seminorm_loss(reconstruction, target)
    if ssim_loss is not None:
        losses["ssim"] = ssim_loss(reconstruction, target)
    return losses


def sds_loss(latent, gamma, teacher, magi_config, shift: float) -> torch.Tensor:
    """Score distillation: MSE between the frozen teacher's velocity and the target (z - noise).

    The teacher's input and output carry no gradient; it flows into the encoder only through z in
    the target. Chunk 0 is clean conditioning (as in DiT training) and is excluded from the loss.
    """
    z = latent.float() * magi_config.runtime_config.scale_factor
    num_chunks = z.shape[2] // CHUNK_WIDTH
    tokens_per_chunk = CHUNK_WIDTH * (z.shape[3] // 2) * (z.shape[4] // 2)

    chunk_times = sample_chunk_t(num_chunks, shift)
    chunk_times[:, 0] = magi_config.runtime_config.clean_t
    noise = torch.randn_like(z)
    noisy = add_noise(z.detach(), chunk_times, noise)
    noisy[:, :, :CHUNK_WIDTH] = z.detach()[:, :, :CHUNK_WIDTH]
    with torch.no_grad():
        teacher_velocity = teacher(noisy, chunk_times, gamma, num_chunks, tokens_per_chunk).float()

    error = (teacher_velocity - (z - noise)).pow(2)[:, :, CHUNK_WIDTH:]
    return error.mean()


def training_micro_step(accelerator, vae, optimizer, trainable_params, target, gamma,
                        ssim_loss, teacher, magi_config, loss_weights, args):
    """Encode -> decode one trajectory (bf16), reconstruction + SDS losses, backward. The optimizer
    only actually steps every grad_accum calls. Returns (dict of detached losses, grad norm or None)."""
    grad_norm = None
    with accelerator.accumulate(vae):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            reconstruction, latent = vae(target)  # through the DDP wrapper, so gradients get synced
        losses = reconstruction_losses(reconstruction, target, ssim_loss, args.recon_loss)
        if teacher is not None:
            losses["sds"] = sds_loss(latent, gamma, teacher, magi_config, args.shift)
        losses["total"] = sum(loss_weights[name] * value for name, value in losses.items())
        accelerator.backward(losses["total"])
        if accelerator.sync_gradients:
            grad_norm = accelerator.clip_grad_norm_(trainable_params, MAX_GRAD_NORM)
        optimizer.step()  # does nothing on accumulation micro-steps
        optimizer.zero_grad(set_to_none=True)
    return {name: value.detach() for name, value in losses.items()}, grad_norm


def all_gpus_out_of_time(start_time: float, max_minutes: float) -> bool:
    """True once any GPU passed the limit; all GPUs must stop on the same step, or DDP hangs."""
    out_of_time = torch.tensor([float(time.time() - start_time > 60 * max_minutes)], device="cuda")
    torch.distributed.all_reduce(out_of_time, op=torch.distributed.ReduceOp.MAX)
    return out_of_time.item() > 0


# ---------------------------------------------------------------- evaluation, logging, saving

@torch.no_grad()
def evaluate(accelerator, vae, eval_set, normalization, step, history_file: Path) -> float:
    """Val VRMSE of the reconstruction, per channel in normalized space, averaged over val sims, then
    over channels (Weihao's checkpoint-selection metric). Computed on GPU 0, shared with all GPUs.
    Each validation is also appended to history_file (<out>/val_history.jsonl), read by utils/vae_sweep.py."""
    result = torch.zeros(1, device="cuda")
    if accelerator.is_main_process:
        dataset, indices = eval_set
        model = accelerator.unwrap_model(vae).magi_vae
        per_sim = []
        for index in indices:
            clip, _ = dataset[index]
            target = normalization.apply(clip[None].float().cuda())
            with torch.autocast("cuda", dtype=torch.bfloat16):
                reconstruction, _ = reconstruct(model, target)
            per_sim.append(vrms_per_channel(reconstruction.float(), target).flatten())
        per_channel = torch.stack(per_sim).mean(dim=0)
        result[0] = per_channel.mean()
        accelerator.print(f"[eval] step {step}  val VRMSE "
                          + "  ".join(f"{name} {value:.4f}" for name, value in zip(CHANNEL_NAMES, per_channel.tolist()))
                          + f"  mean {result.item():.4f}")
        accelerator.print(f"[eval] step {step}  {pressure_path(model)}")
        record = {"step": step, **dict(zip(CHANNEL_NAMES, per_channel.tolist())), "mean": result.item()}
        with open(history_file, "a") as history:
            history.write(json.dumps(record) + "\n")
    torch.distributed.broadcast(result, src=0)
    return result.item()


def pressure_path(vae) -> str:
    """Is pressure really carried? Size of the encoder's pressure input filter relative to the RGB-derived
    ones (the official VAE ended at ~0.06: hardly read), and cosine similarity of the decoder's pressure
    output filter to its density one (official: 0.81, i.e. pressure drawn from density)."""
    if not hasattr(vae.encoder, "patch_embed"):
        return "pressure_path: n/a (adapter VAE)"
    w = vae.encoder.patch_embed.proj.weight.float()
    o = vae.decoder.last_layer.weight.float()
    rms = lambda t: t.pow(2).mean().sqrt()
    size = rms(w[:, 3]) / rms(w[:, :3])
    cosine = torch.nn.functional.cosine_similarity(o[3].flatten(), o[0].flatten(), dim=0)
    return f"pressure_path: encoder pressure filter / RGB filters {size:.3f}   decoder pressure~density cosine {cosine:+.3f}"


def log_step(accelerator, step, recent_losses, grad_norm, scheduler, step_seconds):
    """Mean of each loss term since the last log line (this GPU), grad norm, lr, step time."""
    means = {name: torch.stack([l[name] for l in recent_losses]).mean().item() for name in recent_losses[0]}
    terms = " ".join(f"{name} {value:.5f}" for name, value in means.items() if name != "total")
    accelerator.print(f"step {step:5d}  loss {means['total']:.5f} ({terms})  grad_norm {grad_norm.item():.3f}  "
                      f"lr {scheduler.get_last_lr()[0]:.2e}  step_time {step_seconds:.1f}s")


def save_weights(accelerator, vae, path: Path, saving):
    """io: only the two channel layers in fp32 (~4 MB). full: every VAE weight in bf16 (~1.2 GB).
    adapter: the trained weights (input adapter, decoder, output adapter) in bf16. The metadata
    "vae_mode" tells vae.load_trained_vae which VAE to rebuild, "normalization" which normalization it
    was trained with (read by the DiT trainers and evaluation)."""
    if not accelerator.is_main_process:
        return
    _, mode, normalization, architecture, (extra_channels, refine_channels) = saving
    model = accelerator.unwrap_model(vae).magi_vae  # names like "encoder.patch_embed.proj.weight"
    dtype = torch.float32 if mode == "io" else torch.bfloat16
    keep = {"full": lambda name: True,
            "io": lambda name: name.startswith(IO_LAYERS),
            "adapter": lambda name: name.startswith(("encoder.in_adapter.", "decoder."))}[mode]
    weights = {name: param.detach().to(dtype).cpu().contiguous()
               for name, param in model.named_parameters() if keep(name)}
    metadata = {"vae_mode": architecture, "decoder_extra_channels": str(extra_channels),
                "decoder_refine_channels": str(refine_channels), **normalization.to_metadata()}
    save_file(weights, str(path), metadata=metadata)
    accelerator.print(f"saved {len(weights)} tensors to {path}")


def keep_if_best(accelerator, vae, saving, step: int, val_vrmse: float, state: dict):
    """Overwrites vae_best.safetensors whenever val VRMSE improves; records it in best.json."""
    if val_vrmse >= state["best_val"]:
        return
    state["best_val"], state["best_step"] = val_vrmse, step
    checkpoint_dir = saving[0]
    save_weights(accelerator, vae, checkpoint_dir / "vae_best.safetensors", saving)
    if accelerator.is_main_process:
        (checkpoint_dir / "best.json").write_text(json.dumps({"step": step, "val_vrmse": val_vrmse}))


def save_resume_state(accelerator, vae, optimizer, scheduler, state, checkpoint_dir: Path):
    """Everything an exact continuation needs (utils/resume.py): fp32 trainable weights + optimizer + scheduler
    + counters (step, best val, samples consumed per GPU, GPU count), and every GPU's RNG states."""
    if accelerator.is_main_process:
        model = accelerator.unwrap_model(vae).magi_vae
        torch.save({
            "weights": {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad},
            "optimizer": optimizer.optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "state": {**state, "world_size": accelerator.num_processes},
        }, checkpoint_dir / "resume.pt")
        accelerator.print(f"resume state written to {checkpoint_dir / 'resume.pt'}")
    save_rng_states(checkpoint_dir, accelerator.process_index)
    accelerator.wait_for_everyone()


def load_resume_state(checkpoint_dir: Path, magi_vae, optimizer, scheduler, grad_accum: int) -> dict:
    saved = torch.load(Path(checkpoint_dir) / "resume.pt", map_location="cpu", weights_only=False)
    magi_vae.load_state_dict(saved["weights"], strict=False)
    optimizer.load_state_dict(saved["optimizer"])
    scheduler.load_state_dict(saved["scheduler"])
    state = saved["state"]
    # Resume states from before exact resuming lack these: assume whole accumulation groups, no RNG.
    state.setdefault("micro_batches_done", state["step"] * grad_accum)
    state.setdefault("world_size", None)
    return state


def restore_random_state(accelerator, checkpoint_dir: Path, state: dict) -> None:
    """Continue every GPU's random numbers where the previous job stopped; reseed if that's impossible."""
    restored = restore_rng_states(checkpoint_dir, accelerator.process_index, state["world_size"], accelerator.num_processes)
    flag = torch.tensor([float(restored)], device="cuda")
    torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MIN)  # all GPUs take the same branch
    if flag.item() > 0:
        accelerator.print(f"resumed at step {state['step']} (best val VRMSE {state['best_val']:.4f}): "
                          "same data order and random numbers as an uninterrupted run")
    else:
        set_seed(SEED + state["step"], device_specific=True)
        accelerator.print(f"resumed at step {state['step']} (best val VRMSE {state['best_val']:.4f}); RNG state "
                          "not restorable (older resume state or different GPU count), reseeded")


if __name__ == "__main__":
    run(parse_args())
