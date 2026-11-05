#!/bin/bash
#SBATCH --job-name=lc_download
#SBATCH --account=project_465003416
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/lc/%x_%j.log
# One resolution folder of the euler_mq dataset (rha6696/euler_mq on HuggingFace) next to the 256 data on scratch,
# test subsets first so evaluation can start while the training subsets are still coming. Resumes partial files.
# From the repo root:
#   sbatch jobs/lumi/lc/download_euler_mq.sbatch 512x512_orig
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
res=${1:?resolution folder, e.g. 512x512_orig}
dest=/scratch/project_465003416/pretrained_video_datasets/euler_mq
hf=/project/project_465003416/venvs/windinet-editor/bin/hf
export HF_HUB_DISABLE_PROGRESS_BARS=1
for part in test train; do
    "$hf" download rha6696/euler_mq --repo-type dataset --local-dir "$dest" \
        --include "$res/$part.h5" --include "$res/${part}_subsets/*"
    echo "$part done $(date)"
done
"$hf" download rha6696/euler_mq --repo-type dataset --local-dir "$dest" --include "$res/*"  # whatever else the folder holds
du -sh "$dest/$res" && ls -la "$dest/$res"
