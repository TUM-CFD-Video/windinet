"""Full finetuning of MAGI-1 4.5B on the Euler dataset.

What is trained:   every DiT weight + the gamma embedding.
What is frozen:    the VAE.

Settings match Weihao's LTX DiT runs (settings.FULL_FT) so results are comparable; the one deviation
is the optimizer (8-bit AdamW from bitsandbytes, to fit in GPU memory). Memory-saving setup:
fp32 master weights computed in bf16 (autocast), gradient checkpointing, gradients stored directly
in DDP's communication buckets.

Read `run()` first; every helper below it does one thing.
"""

import os
import time
from pathlib import Path

import torch
import yaml
from accelerate.utils import DistributedDataParallelKwargs, set_seed
from safetensors.torch import save_file

import magi_finetuning.compat as compat
from magi_finetuning.data import Normalization, batch_to_latents, make_loader
from magi_finetuning.dit import (Model, add_noise, enable_gradient_checkpointing, gamma_embedding, keep_image_clean,
                                 load_dit, sample_chunk_t, sample_num_clean_chunks, velocity_loss)
from magi_finetuning.settings import FULL_FT
from magi_finetuning.utils.resume import batches_from, restore_rng_states, save_rng_states
from magi_finetuning.utils.runtime import JobTimer, check_adamw8bit, check_ckpt_dir, init_distributed, shutdown
from magi_finetuning.vae import load_vae_encoder


def run(args):
    timer = JobTimer()  # setup / training time -> <ckpt-dir>/timing.json
    # 1. Multi-GPU setup: effective batch 32 = 1 sample per GPU x grad_accum x number of GPUs
    num_gpus = int(os.environ.get("WORLD_SIZE", "1"))
    grad_accum = gradient_accumulation_for(num_gpus)
    accelerator = init_distributed(grad_accum, [DistributedDataParallelKwargs(gradient_as_bucket_view=True)])
    set_seed(FULL_FT["seed"], device_specific=True)  # Weihao's base seed; different noise on every GPU

    # 2. Fail fast: fused attention kernels only, and 8-bit AdamW must really work on this GPU
    restrict_attention_to_fused_kernels(accelerator)
    check_adamw8bit()
    accelerator.print("[check] bitsandbytes AdamW8bit: real 8-bit optimizer step on the GPU OK")
    prepare_output_dir(args, accelerator, grad_accum, num_gpus)  # writes <out>/args.yaml

    # 3. Model, checkpoint folder, frozen VAE encoder, data
    model, magi_config, trainable_params = build_full_model(args)
    num_params = sum(p.numel() for p in trainable_params)
    accelerator.print(f"trainable params: {num_params:,} (all DiT + gamma embedding)")
    checkpoint_dir = prepare_checkpoint_dir(args, accelerator, num_params)  # None with --no-save
    vae_encoder = load_vae_encoder(args.vae_ckpt)
    normalization = Normalization.of_vae_checkpoint(args.vae_ckpt)  # the one the VAE was trained with
    accelerator.print(f"normalization: {normalization.describe()}")
    loader, sampler = make_loader(args, seed=FULL_FT["seed"])

    # 4. Optimizer and learning-rate schedule (cosine to eta_min, no warmup)
    optimizer = make_adamw8bit(trainable_params, args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=FULL_FT["eta_min"])
    # Continue a run that stopped at --max-minutes (8000 steps don't fit one 12 h job), if there is one.
    progress = {"step": 0, "micro_batches_done": 0, "world_size": None}
    if args.resume and (checkpoint_dir_for_resume(args) / "resume.pt").is_file():
        progress = load_resume_state(checkpoint_dir_for_resume(args), model, optimizer, scheduler, grad_accum)
        if progress["step"] >= args.steps:  # a spare job in the chain after the run already finished
            accelerator.print(f"run already finished at step {progress['step']}; exiting")
            shutdown(accelerator)
            return
    elif args.init_from:
        load_initial_weights(model, args.init_from)
        accelerator.print(f"new run from the weights of {args.init_from} (fresh optimizer, lr schedule from step 0)")
    step = progress["step"]
    # prepare() wraps the model for multi-GPU and splits the data across GPUs. The scheduler is
    # deliberately NOT passed in: Accelerate would step it once per GPU, i.e. too fast.
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    # Same data order as one long job: rebuild the epoch, skip what this GPU already saw.
    batches = batches_from(loader, accelerator, progress["micro_batches_done"], epoch_aware=[sampler])
    micro_batches_done = progress["micro_batches_done"]
    if step > 0:
        restore_random_state(accelerator, checkpoint_dir_for_resume(args), progress["world_size"], step)

    # 5. Training loop
    recent_losses = []
    steps_from = step
    timer.training_started()
    start_time = last_step_time = time.time()
    for batch in batches:
        micro_batches_done += 1
        # frame 0 -> clean image latent; frames 1..F -> body latents (data.build_latent_sequence)
        clean_latents, gamma, num_body, num_chunks, tokens_per_chunk = batch_to_latents(
            batch, vae_encoder, normalization, magi_config.runtime_config.scale_factor
        )
        noisy_latents, chunk_times, noise, num_clean = make_noisy_input(clean_latents, num_chunks, args)

        loss, grad_norm = training_micro_step(
            accelerator, model, optimizer, trainable_params,
            noisy_latents, chunk_times, gamma, num_chunks, tokens_per_chunk, clean_latents, noise, num_body, num_clean,
        )
        recent_losses.append(loss)

        # Only every grad_accum-th micro-step is a real optimizer step.
        if not accelerator.sync_gradients:
            continue
        learning_rate = scheduler.get_last_lr()[0]  # the lr this step was taken with
        scheduler.step()
        step += 1

        now = time.time()
        log_step(accelerator, step, recent_losses, grad_norm, learning_rate, now - last_step_time)
        recent_losses = []
        last_step_time = now

        if checkpoint_dir is not None and (step == args.steps or step % args.save_every == 0):
            save_full_checkpoint(accelerator, model, optimizer, checkpoint_dir, step, args.save_optimizer)
        if step >= args.steps:
            break
        if args.max_minutes is not None and all_gpus_out_of_time(start_time, args.max_minutes):
            accelerator.print(f"--max-minutes {args.max_minutes} reached after {step} steps; stopping")
            break

    # 6. Resume state (also when finished, so a later job sees "done" instead of an older step),
    #    memory and speed report
    if not args.no_save:  # every GPU enters: save_resume_state ends with a sync
        save_resume_state(accelerator, model, optimizer, scheduler, step, micro_batches_done, Path(args.ckpt_dir))
    report_memory_and_speed(accelerator, step - steps_from, start_time)  # this job's steps only (resumed jobs)
    if accelerator.is_main_process:  # next to the checkpoints, or the --out dir with --no-save
        timing_dir = Path(args.out) if args.no_save else Path(args.ckpt_dir)
        timer.record(timing_dir / "timing.json", "dit", steps_from, step, num_gpus)
    shutdown(accelerator)


# ---------------------------------------------------------------- setup

def gradient_accumulation_for(num_gpus: int) -> int:
    """Micro-steps per optimizer step so that 1 sample/GPU x num_gpus x result = effective batch (32)."""
    effective_batch = FULL_FT["effective_batch"]
    if effective_batch % num_gpus:
        raise ValueError(f"effective batch {effective_batch} not divisible by {num_gpus} GPUs")
    return effective_batch // num_gpus


def restrict_attention_to_fused_kernels(accelerator) -> None:
    """Attention must use a fused (fast) kernel; anything else raises instead of silently slowing down.
    GPU 0 also prints which fused kernels are available."""
    if accelerator.is_main_process:
        compat.restrict_sdpa_backends()
    else:
        compat._restrict_backends = True


def prepare_output_dir(args, accelerator, grad_accum: int, num_gpus: int) -> Path:
    """Records the run's arguments and the fixed full-FT settings in <out>/args.yaml."""
    output_dir = Path(args.out)
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        settings = {**FULL_FT, "lr": args.lr, "betas": list(FULL_FT["betas"])}  # yaml can't write tuples
        record = yaml.safe_dump({**vars(args), "full_ft_settings": settings, "grad_accum": grad_accum, "num_gpus": num_gpus})
        (output_dir / "args.yaml").write_text(record)
        if args.ckpt_dir and not args.no_save:  # evaluation reads it next to the checkpoints
            Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
            (Path(args.ckpt_dir) / "args.yaml").write_text(record)
    return output_dir


def build_full_model(args):
    """MAGI-1 DiT with every weight trainable in fp32 (computation still runs in bf16), plus the gamma embedding.

    Returns (model, magi_config, trainable_params).
    """
    dit, magi_config = load_dit()
    compat.set_inference_kv_range(magi_config, args.inference_kv_range)  # only recorded, used when sampling
    dit = dit.float().requires_grad_(True)  # fp32 "master" weights so small updates aren't lost to bf16 rounding
    enable_gradient_checkpointing(dit)      # saves GPU memory: recompute activations in backward
    model = Model(dit, gamma_embedding())
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    return model, magi_config, trainable_params


def prepare_checkpoint_dir(args, accelerator, num_params: int):
    """None with --no-save. Otherwise --ckpt-dir is required, checked (outside home/IG_work, enough
    free space for every planned save) on GPU 0, and returned there; other GPUs get None (they never save)."""
    if args.no_save:
        return None
    if not args.ckpt_dir:
        raise ValueError("--full-finetune needs --ckpt-dir (no default) or --no-save")
    if not accelerator.is_main_process:
        return None
    save_steps = set(range(args.save_every, args.steps + 1, args.save_every)) | {args.steps}
    return check_ckpt_dir(args, num_params, len(save_steps))


def make_adamw8bit(trainable_params, lr: float):
    """bitsandbytes 8-bit AdamW with Weihao's betas / weight decay; lr = --lr (default Weihao's FULL_FT lr)."""
    import bitsandbytes as bnb

    return bnb.optim.AdamW8bit(
        trainable_params, lr=lr, betas=FULL_FT["betas"], weight_decay=FULL_FT["weight_decay"]
    )


# ---------------------------------------------------------------- one training step

def make_noisy_input(clean_latents, num_chunks, args):
    """Noise every chunk to its own random level t (0 = pure noise, 1 = clean); earlier chunks cleaner.
    With --clean-chunks the first few chunks may be clean context (dit.sample_num_clean_chunks).
    Latent frame 0 (the image = given frame 0) is then put back clean, as MAGI-1's image-to-video does
    at every sampling step. Returns (noisy_latents, chunk_times, noise, number of clean chunks).
    """
    num_clean = sample_num_clean_chunks(num_chunks) if args.clean_chunks else 0
    chunk_times = sample_chunk_t(num_chunks, args.shift, num_clean)  # shape (1, num_chunks)
    noise = torch.randn_like(clean_latents)
    noisy_latents = keep_image_clean(add_noise(clean_latents, chunk_times, noise), clean_latents)
    return noisy_latents, chunk_times, noise, num_clean


def training_micro_step(accelerator, model, optimizer, trainable_params,
                        noisy_latents, chunk_times, gamma, num_chunks, tokens_per_chunk,
                        clean_latents, noise, num_body: int, num_clean: int):
    """Forward (in bf16) + backward on one sample. The optimizer only actually steps every grad_accum calls.

    Returns (loss tensor, gradient norm or None). The gradient norm exists only on real optimizer steps.
    """
    grad_norm = None

    # accumulate() skips the cross-GPU gradient sync on all but the last micro-step.
    with accelerator.accumulate(model):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            predicted_velocity = model(noisy_latents, chunk_times, gamma, num_chunks, tokens_per_chunk)
        # Target is (clean - noise), scored on frames 1..F of the noisy chunks only (not the given image,
        # not the padding, not clean context chunks).
        loss = velocity_loss(predicted_velocity, clean_latents, noise, num_body, num_clean)
        accelerator.backward(loss)

        if accelerator.sync_gradients:
            grad_norm = accelerator.clip_grad_norm_(trainable_params, FULL_FT["max_grad_norm"])
        optimizer.step()  # does nothing on accumulation micro-steps
        # set_to_none=False: the gradients live inside DDP's buckets (gradient_as_bucket_view); keep them there.
        optimizer.zero_grad(set_to_none=False)
    return loss.detach(), grad_norm


def all_gpus_out_of_time(start_time: float, max_minutes: float) -> bool:
    """True once any GPU has passed the time limit. All GPUs must stop on the same step, or DDP hangs."""
    out_of_time = torch.tensor([float(time.time() - start_time > 60 * max_minutes)], device="cuda")
    torch.distributed.all_reduce(out_of_time, op=torch.distributed.ReduceOp.MAX)
    return out_of_time.item() > 0


# ---------------------------------------------------------------- logging and saving

def log_step(accelerator, step, recent_losses, grad_norm, learning_rate, step_seconds):
    """Every step: loss (mean over this step's micro-steps and all GPUs), grad norm, lr, step time."""
    this_gpu_loss = torch.stack(recent_losses).mean()
    mean_loss = accelerator.gather(this_gpu_loss[None]).mean().item()
    accelerator.print(
        f"step {step:4d}  loss {mean_loss:.4f}  grad_norm {grad_norm.item():.3f}  "
        f"lr {learning_rate:.3e}  step_time {step_seconds:.1f}s"
    )


def save_full_checkpoint(accelerator, model, optimizer, checkpoint_dir, step, save_optimizer: bool):
    """All weights in bf16 (weights only); the 8-bit optimizer state only with --save-optimizer."""
    unwrapped_model = accelerator.unwrap_model(model)
    weights = {
        name: param.detach().to(torch.bfloat16).cpu().contiguous()
        for name, param in unwrapped_model.named_parameters()
    }
    save_file(weights, str(checkpoint_dir / f"full_step{step:06d}.safetensors"))
    if save_optimizer:
        torch.save(optimizer.optimizer.state_dict(), checkpoint_dir / f"optimizer_step{step:06d}.pt")
    accelerator.print(f"saved bf16 weights{' + optimizer' if save_optimizer else ''} at step {step}")


def checkpoint_dir_for_resume(args) -> Path:
    if not args.ckpt_dir:
        raise ValueError("--resume needs the --ckpt-dir of the run to continue")
    return Path(args.ckpt_dir)


def save_resume_state(accelerator, model, optimizer, scheduler, step: int, micro_batches_done: int,
                      checkpoint_dir: Path):
    """Everything an exact continuation needs (utils/resume.py), overwritten each time, on SCRATCH:
    fp32 master weights + 8-bit AdamW state + scheduler + step + samples consumed per GPU (~27 GB),
    plus every GPU's RNG states (rng_rank<r>.pt). Separate from the bf16 checkpoints, which are for
    evaluation and lose the fp32 precision."""
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        torch.save({
            "weights": {name: param.detach().cpu() for name, param in unwrapped.named_parameters()},
            "optimizer": optimizer.optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
            "micro_batches_done": micro_batches_done,
            "world_size": accelerator.num_processes,
        }, checkpoint_dir / "resume.pt")
        accelerator.print(f"resume state for step {step} written to {checkpoint_dir / 'resume.pt'}")
    save_rng_states(checkpoint_dir, accelerator.process_index)
    accelerator.wait_for_everyone()


def load_initial_weights(model, checkpoint: str) -> None:
    """--init-from: the weights of another run, from its resume.pt (fp32 master weights) or a
    full_stepNNNNNN.safetensors (bf16, cast to the fp32 parameters). Every parameter must be there."""
    if checkpoint.endswith(".pt"):
        weights = torch.load(checkpoint, map_location="cpu", weights_only=False)["weights"]
    else:
        from safetensors.torch import load_file
        weights = load_file(checkpoint)
    missing, unexpected = model.load_state_dict(weights, strict=False)
    parameter_names = {name for name, _ in model.named_parameters()}
    if unexpected or any(name in parameter_names for name in missing):
        raise RuntimeError(f"{checkpoint} doesn't match the model: missing {missing[:5]}, unexpected {unexpected[:5]}")


def load_resume_state(checkpoint_dir: Path, model, optimizer, scheduler, grad_accum: int) -> dict:
    """Loads weights, optimizer, scheduler; returns step, samples consumed per GPU and the GPU count."""
    saved = torch.load(checkpoint_dir / "resume.pt", map_location="cpu", weights_only=False)
    # Only parameters are saved; buffers (the gamma embedding's fixed ranges and Fourier frequencies)
    # are rebuilt from the config, so they may be missing, but no parameter may be.
    missing, unexpected = model.load_state_dict(saved["weights"], strict=False)
    parameter_names = {name for name, _ in model.named_parameters()}
    if unexpected or any(name in parameter_names for name in missing):
        raise RuntimeError(f"resume.pt doesn't match the model: missing {missing[:5]}, unexpected {unexpected[:5]}")
    optimizer.load_state_dict(saved["optimizer"])
    scheduler.load_state_dict(saved["scheduler"])
    step = saved["step"]
    # Resume states from before exact resuming have no sample count: assume whole accumulation groups.
    return {"step": step, "micro_batches_done": saved.get("micro_batches_done", step * grad_accum),
            "world_size": saved.get("world_size")}


def restore_random_state(accelerator, checkpoint_dir: Path, saved_world_size, step: int) -> None:
    """Continue every GPU's random numbers where the previous job stopped; reseed if that's impossible."""
    restored = restore_rng_states(checkpoint_dir, accelerator.process_index, saved_world_size, accelerator.num_processes)
    if not all_gpus_agree(restored):
        set_seed(FULL_FT["seed"] + step, device_specific=True)
        accelerator.print(f"resumed at step {step}; RNG state not restorable (older state or different GPU count), reseeded")
    else:
        accelerator.print(f"resumed at step {step}: same data order and random numbers as an uninterrupted run")


def all_gpus_agree(flag: bool) -> bool:
    """True only if flag is True on every GPU (so all GPUs take the same branch)."""
    value = torch.tensor([float(flag)], device="cuda")
    torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.MIN)
    return value.item() > 0


def report_memory_and_speed(accelerator, steps_done: int, start_time: float):
    """Peak GPU memory of every GPU, and the mean time per optimizer step."""
    my_peak_gb = torch.tensor([torch.cuda.max_memory_allocated() / 1e9], device="cuda")
    peak_per_gpu = accelerator.gather(my_peak_gb).tolist()
    accelerator.print("max_memory_allocated per rank (GB): "
                      + ", ".join(f"rank{rank}={gb:.1f}" for rank, gb in enumerate(peak_per_gpu)))
    mean_step_seconds = (time.time() - start_time) / max(steps_done, 1)
    accelerator.print(f"steps done: {steps_done}, mean step time {mean_step_seconds:.1f}s (incl. first-step warmup)")
