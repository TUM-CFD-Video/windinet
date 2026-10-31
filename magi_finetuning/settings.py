"""Paths and fixed hyperparameters."""

from pathlib import Path

from magi_finetuning.compat import IG_WORK

REPO = Path(__file__).resolve().parents[1]
# MAGI-1 weights on project storage (approved by Harish; scratch is for training checkpoints).
MAGI1_WEIGHTS = Path("/e/project1/e-dev-2026d09-262/ig_work/magi1_pretrained")
CKPT = MAGI1_WEIGHTS / "ckpt"
MAGI_CONFIG = IG_WORK / "MAGI-1" / "example" / "4.5B" / "4.5B_base_config.json"

CHUNK_WIDTH = 6  # latent frames per chunk (= 24 real frames)
SAMPLER_WINDOW = 4  # chunks denoised at once at inference (MAGI-1's runtime window_size; dit.rollout)
# Conditioning: frame 0 is given (as MAGI-1's image-to-video), frames 1..F are generated.
# F = 100 matches Weihao's LTX runs (all 101 frames of a trajectory); see data.build_latent_sequence.
DEFAULT_FRAMES = 100

# Channel normalization for every MAGI path (VAE, LoRA, full FT): x -> clamp((x - mean) / (CLIP * std), -1, 1),
# per channel, with the train-split stats the LTX 256x256 runs use. Fixed here so the paths can't diverge.
NORMALIZATION_STATS = REPO / "configs" / "finetune_vae" / "euler_mq_256_only_train.yaml"
NORMALIZATION_CLIP = 5.0

# Outputs (results, args, evaluations, videos) and Slurm logs live on SCRATCH, never in the repo (code only);
# same paths as jobs/jupiter/magi1/magi1_paths.sh.
SCRATCH = Path("/e/scratch/e-dev-2026d09-262/ig_work")
OUTPUTS = SCRATCH / "magi1_outputs"

# Final models: every completed evaluation copies its VAE + DiT checkpoints and results here (eval/evaluate.py
# archive_run). Storage rule: PROJECT holds only pretrained models and datasets; outputs, checkpoints and final
# models all live on SCRATCH.
# Also the "meta folder" that jobs/jupiter/magi1/evaluate_experiments.sh evaluates in one go.
FINAL_MODELS = SCRATCH / "magi1_final"

# Data. train.h5 is split 90/10 into train/val once (the table SPLIT_FILE, see data.py); test.h5 is for inference only.
TRAIN_DATA = "/e/project1/e-dev-2026d09-262/datasets/euler_mq_dataset/256x256_ds/train.h5"
TEST_DATA = "/e/project1/e-dev-2026d09-262/datasets/euler_mq_dataset/256x256_ds/test.h5"
SPLIT_FILE = REPO / "configs" / "magi1" / "train_val_split_256.csv"  # one row per sim: sim_id, gamma, split
# Not linear_kv_xattn: MAGI's get_xqkv reads its .weight directly (dit_module.py:1012) instead of calling the
# module, so a LoRA adapter there is never used (and DDP fails on its unused params). The cross-attention K/V
# still adapt through the gamma embedding feeding them, which is fully trained.
LORA_TARGETS = ["linear_qkv.q", "linear_qkv.qx", "linear_qkv.k", "linear_qkv.v", "linear_proj"]
#q, k, v, qx, linear_proj
# Weihao's DiT settings (configs/dit/train_dit_sng_pvc_baseline.yaml + torch AdamW defaults), for comparability.
FULL_FT = dict(
    lr=1e-5, weight_decay=0.01, betas=(0.9, 0.999), eta_min=1e-8, warmup=0, max_grad_norm=1.0,
    effective_batch=32, seed=42,
    optimizer="bitsandbytes AdamW8bit -- the only deviation (Weihao: torch AdamW); fp32 master weights",
)
