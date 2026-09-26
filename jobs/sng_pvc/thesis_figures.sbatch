#!/bin/bash
#SBATCH --job-name=wh_thesis_figures
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=general
#SBATCH --account=pn82ku
#SBATCH -o logs/sng_pvc/%j-%x.out
#SBATCH -e logs/sng_pvc/%j-%x.err
# Same broken-scratch-mount node jobs/sng_pvc/latent_stats.sbatch excludes.
#SBATCH --exclude=i20r02c04s06

# Renders the thesis field figures (scripts/thesis_figures.py) from one DiT
# eval sample npz on scratch. The plotting is CPU-only matplotlib; it runs as a
# batch job only because the login nodes have no python / windinet env.
#
# Both figures read figure_data/dit/<NPZ_SUBDIR>/<SIM>.npz (gt_raw, gt_norm,
# vae_only_norm, vae_dit_norm at frames 0/25/50/75/100):
#   dataset        Ch3  raw fields at t = 0/50/100
#   vae-dit-frame  Ch6  baseline GT | VAE only | VAE+DiT | |error| x2 at frame 100
#
# Default: sim 2507 (gamma 1.40), a TRAINING sim picked for its rich structure
# (say so in the caption). Its npz comes from a one-sim baseline DiT eval:
#   R=/hppfs/scratch/0D/go76fuz2/windinet
#   SIM_IDS=2507_gamma1.3999999762 NPZ_DIR=$R/figure_data/dit/loss_rmse_h1_picked \
#     sbatch jobs/sng_pvc/eval_dit_vrmse.sbatch \
#       $R/dit_preprocessed/finetune_vae_ch6_loss_rmse_h1_256res \
#       $R/outputs/shockwave_dit_ch6_loss_rmse_h1/checkpoints/model_weights_step_08000.safetensors \
#       $R/finetune_vae_outputs_sng_pvc/finetune_vae_ch6_loss_rmse_h1_256res/checkpoints/vae_shockwave_best.safetensors \
#       1 0
# The test-set sims from the Chapter 6 DiT eval work too:
#   SIM=0250_gamma1.3999999762 NPZ_SUBDIR=loss_rmse_h1 sbatch jobs/sng_pvc/thesis_figures.sbatch
#
# The PNGs land in figures/ of this checkout; commit and push them from here.
#
# Usage:
#   sbatch [--dependency=afterok:<eval_job>] jobs/sng_pvc/thesis_figures.sbatch

set -eo pipefail

module load micromamba/2.1.1
export PATH="$MICROMAMBA_BASE/bin:$PATH"
eval "$(micromamba shell hook -s bash)"
micromamba activate windinet

cd /dss/dsshome1/0D/go76fuz2/windinet

echo "START TIME: $(date)"
echo "job=${SLURM_JOB_ID} node=$(hostname)"

SIM=${SIM:-2507_gamma1.3999999762}
NPZ_SUBDIR=${NPZ_SUBDIR:-loss_rmse_h1_picked}
NPZ="${SCRATCH}/windinet/figure_data/dit/${NPZ_SUBDIR}/${SIM}.npz"
echo "SIM=${SIM} NPZ=${NPZ}"
[ -f "${NPZ}" ] || { echo "missing ${NPZ} -- run the one-sim DiT eval first (see header)" >&2; exit 1; }

python scripts/thesis_figures.py dataset --npz "${NPZ}" \
    --frames 0 --frames 50 --frames 100 \
    --out "figures/ch3_euler_mq_${SIM%%_*}.png"
python scripts/thesis_figures.py vae-dit-frame --npz "${NPZ}" --frame 100 \
    --out "figures/ch6_vae_dit_frame100_${SIM%%_*}.png"

echo "END TIME: $(date)"
