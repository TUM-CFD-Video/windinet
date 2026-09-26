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

# Renders the thesis field figures (scripts/thesis_figures.py) from the eval
# sample npz files on scratch. The plotting is CPU-only matplotlib; it runs as a
# batch job only because the login nodes have no python / windinet env.
#
#   dataset        <- figure_data/vae_test/finetune_vae_ch6_loss_rmse_h1_256res/<SIM>.npz
#                     (written by the VAE test eval; frames 0/25/50/75/100)
#   vae-dit-frame  <- figure_data/dit/loss_rmse_h1/<SIM>.npz
#                     (written by the baseline DiT eval; skipped until it exists)
#
# The PDFs land in figures/ of this checkout; commit and push them from here.
# SIM picks one of the three gamma-spread test sims (0000/0250/0499); the
# default is the median-gamma one.
#
# Usage:
#   sbatch jobs/sng_pvc/thesis_figures.sbatch
#   SIM=0000_gamma1.1299999952 sbatch jobs/sng_pvc/thesis_figures.sbatch

set -eo pipefail

module load micromamba/2.1.1
export PATH="$MICROMAMBA_BASE/bin:$PATH"
eval "$(micromamba shell hook -s bash)"
micromamba activate windinet

cd /dss/dsshome1/0D/go76fuz2/windinet

echo "START TIME: $(date)"
echo "job=${SLURM_JOB_ID} node=$(hostname)"

SIM=${SIM:-0250_gamma1.3999999762}
FIG_DATA="${SCRATCH}/windinet/figure_data"
VAE_NPZ="${FIG_DATA}/vae_test/finetune_vae_ch6_loss_rmse_h1_256res/${SIM}.npz"
DIT_NPZ="${FIG_DATA}/dit/loss_rmse_h1/${SIM}.npz"
echo "SIM=${SIM}"

if [ -f "${VAE_NPZ}" ]; then
    python scripts/thesis_figures.py dataset --npz "${VAE_NPZ}" \
        --frames 0 --frames 50 --frames 100 \
        --out "figures/ch3_euler_mq_${SIM%%_*}.pdf"
else
    echo "missing ${VAE_NPZ} -- dataset figure skipped" >&2
fi

if [ -f "${DIT_NPZ}" ]; then
    python scripts/thesis_figures.py vae-dit-frame --npz "${DIT_NPZ}" --frame 100 \
        --out "figures/ch6_vae_dit_frame100_${SIM%%_*}.pdf"
else
    echo "missing ${DIT_NPZ} -- vae-dit-frame figure skipped (baseline DiT eval not done yet?)" >&2
fi

echo "END TIME: $(date)"
