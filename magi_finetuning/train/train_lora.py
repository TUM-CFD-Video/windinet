"""LoRA finetuning of MAGI-1 4.5B on the Euler dataset.

What is trained:   rank-32 LoRA adapters on the attention layers + the gamma embedding.
What is frozen:    every original MAGI-1 weight, and the VAE.

One optimizer step = `grad_accum` samples per GPU (one sample at a time), gradients averaged
over all GPUs; by default grad_accum = 32 / number of GPUs (effective batch 32 on any GPU count).
LoRA checkpoints (~150 MB each) go to --ckpt-dir (SCRATCH), or <out>/checkpoints without it.
Read `run()` first; every helper below it does one thing.
"""

import os
from pathlib import Path

import torch
import yaml
from peft import LoraConfig, inject_adapter_in_model
from safetensors.torch import save_file

from magi_finetuning.data import Normalization, batch_to_latents, make_loader
from magi_finetuning.dit import (Model, add_noise, enable_gradient_checkpointing, gamma_embedding, keep_image_clean,
                                 load_dit, sample_chunk_t, sample_num_clean_chunks, velocity_loss)
from magi_finetuning.settings import LORA_TARGETS
from magi_finetuning.utils.resume import batches_from
from magi_finetuning.utils.runtime import JobTimer, init_distributed, shutdown
from magi_finetuning.vae import load_vae_encoder

MAX_GRAD_NORM = 1.0
EFFECTIVE_BATCH = 32
SEED = 1234


def run(args):
    timer = JobTimer()  # setup / training time -> <checkpoint dir>/timing.json
    # 1. Multi-GPU setup and output folders
    grad_accum = args.grad_accum or EFFECTIVE_BATCH // int(os.environ.get("WORLD_SIZE", "1"))
    accelerator = init_distributed(grad_accum)
    torch.manual_seed(SEED + accelerator.process_index)  # different noise on every GPU
    checkpoint_dir = prepare_output_dirs(args, accelerator, grad_accum)

    # 2. Model, frozen VAE encoder, data
    model, magi_config, trainable_params = build_lora_model(args)
    accelerator.print(f"trainable params: {sum(p.numel() for p in trainable_params):,}")
    vae_encoder = load_vae_encoder(args.vae_ckpt)
    normalization = Normalization.of_vae_checkpoint(args.vae_ckpt)  # the one the VAE was trained with
    accelerator.print(f"normalization: {normalization.describe()}")
    loader, sampler = make_loader(args, seed=SEED)

    # 3. Optimizer and learning-rate schedule
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.0)
    scheduler = linear_warmup_then_constant(optimizer, args.warmup)
    # prepare() wraps the model for multi-GPU and splits the data across GPUs. The scheduler is
    # deliberately NOT passed in: Accelerate would step it once per GPU, i.e. too fast.
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)

    # 4. Training loop (a new shuffle every epoch, as in train_full)
    step = 0
    recent_losses = []
    timer.training_started()
    for batch in batches_from(loader, accelerator, 0, epoch_aware=[sampler]):
        # frame 0 -> clean image latent; frames 1..F -> body latents (data.build_latent_sequence)
        clean_latents, gamma, num_body, num_chunks, tokens_per_chunk = batch_to_latents(
            batch, vae_encoder, normalization, magi_config.runtime_config.scale_factor
        )
        noisy_latents, chunk_times, noise, num_clean = make_noisy_input(clean_latents, num_chunks, args)

        loss = training_micro_step(
            accelerator, model, optimizer, trainable_params,
            noisy_latents, chunk_times, gamma, num_chunks, tokens_per_chunk, clean_latents, noise, num_body, num_clean,
        )
        recent_losses.append(loss)

        # Only every grad_accum-th micro-step is a real optimizer step.
        if not accelerator.sync_gradients:
            continue
        scheduler.step()
        step += 1

        if step == 1 or step % args.log_every == 0:
            log_progress(accelerator, step, recent_losses, scheduler)
            recent_losses = []
        if step == args.steps or step % args.save_every == 0:
            save_lora_checkpoint(accelerator, model, checkpoint_dir, step)
        if step >= args.steps:
            break

    if accelerator.is_main_process:
        timer.record(checkpoint_dir / "timing.json", "dit_lora", 0, step, accelerator.num_processes)
    shutdown(accelerator)


# ---------------------------------------------------------------- setup

def prepare_output_dirs(args, accelerator, grad_accum: int) -> Path:
    """Creates the checkpoint folder (--ckpt-dir, or <out>/checkpoints) and records the run's arguments
    in args.yaml in both <out> and the checkpoint folder (evaluation reads it next to the checkpoints).
    Returns the checkpoint folder."""
    output_dir = Path(args.out)
    checkpoint_dir = Path(args.ckpt_dir) if args.ckpt_dir else output_dir / "checkpoints"
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        record = yaml.safe_dump({**vars(args), "grad_accum": grad_accum, "num_gpus": accelerator.num_processes})
        for folder in (output_dir, checkpoint_dir):
            (folder / "args.yaml").write_text(record)
    return checkpoint_dir


def build_lora_model(args):
    """MAGI-1 DiT with all original weights frozen, LoRA adapters added, plus the gamma embedding.

    Returns (model, magi_config, trainable_params).
    """
    dit, magi_config = load_dit()
    dit.requires_grad_(False)

    lora_config = LoraConfig(r=args.lora_rank, lora_alpha=args.lora_rank, target_modules=LORA_TARGETS)
    inject_adapter_in_model(lora_config, dit)

    # The adapters are the only DiT weights we train. Keep them in fp32 for stable updates
    # (the frozen base weights stay in MAGI's own bf16/fp32 mix).
    for name, param in dit.named_parameters():
        if "lora_" in name:
            param.data = param.data.float()
            param.requires_grad_(True)

    enable_gradient_checkpointing(dit)  # saves GPU memory: recompute activations in backward
    model = Model(dit, gamma_embedding())
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    return model, magi_config, trainable_params


def linear_warmup_then_constant(optimizer, warmup_steps: int):
    """Learning rate ramps linearly from lr/warmup to lr over `warmup_steps` steps, then stays."""
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1.0, (step + 1) / warmup_steps))


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
                        clean_latents, noise, num_body: int, num_clean: int) -> float:
    """Forward + backward on one sample. The optimizer only actually steps every grad_accum calls."""
    # accumulate() skips the cross-GPU gradient sync on all but the last micro-step.
    with accelerator.accumulate(model):
        predicted_velocity = model(noisy_latents, chunk_times, gamma, num_chunks, tokens_per_chunk)
        # Target is (clean - noise), scored on frames 1..F of the noisy chunks only (not the given image,
        # not the padding, not clean context chunks).
        loss = velocity_loss(predicted_velocity, clean_latents, noise, num_body, num_clean)
        accelerator.backward(loss)

        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(trainable_params, MAX_GRAD_NORM)
        optimizer.step()       # does nothing on accumulation micro-steps
        optimizer.zero_grad(set_to_none=True)
    return loss.item()


# ---------------------------------------------------------------- logging and saving

def log_progress(accelerator, step, recent_losses, scheduler):
    """Mean loss since the last log line (this GPU only), current lr, peak GPU memory."""
    mean_loss = sum(recent_losses) / len(recent_losses)
    learning_rate = scheduler.get_last_lr()[0]
    peak_memory_gb = torch.cuda.max_memory_allocated() / 1e9
    accelerator.print(f"step {step}  loss {mean_loss:.4f}  lr {learning_rate:.2e}  mem {peak_memory_gb:.1f}GB")


def save_lora_checkpoint(accelerator, model, checkpoint_dir: Path, step):
    """Saves only the trained weights (LoRA adapters + gamma embedding), in fp32, from GPU 0."""
    if not accelerator.is_main_process:
        return
    unwrapped_model = accelerator.unwrap_model(model)
    trained_weights = {
        name: param.detach().float().cpu().contiguous()
        for name, param in unwrapped_model.named_parameters()
        if param.requires_grad
    }
    path = checkpoint_dir / f"lora_step{step:06d}.safetensors"
    save_file(trained_weights, str(path))
    accelerator.print(f"saved checkpoint at step {step}")
