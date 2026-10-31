# Where MAGI-1 jobs put things -- everything on SCRATCH, final models included (purged after 90 days untouched; every training job
# touches it). The repo holds code only. Sourced by magi1_env.sh (jobs) and the submit scripts (login node).
# The #SBATCH --output lines can't use variables and repeat MAGI1_LOGS literally.
export MAGI1_SCRATCH="/e/scratch/e-dev-2026d09-262/ig_work"
export MAGI1_OUTPUTS="${MAGI1_SCRATCH}/magi1_outputs"     # results, args, evaluations, videos, STATUS.md
export MAGI1_LOGS="${MAGI1_SCRATCH}/logs"                 # Slurm logs (%x_%j.log / .err) and shard logs
export MAGI1_OFFICIAL="${MAGI1_SCRATCH}/magi1_official"   # checkpoints of the first official run
export MAGI1_FINAL="${MAGI1_SCRATCH}/magi1_final"         # final models of completed evaluations (settings.FINAL_MODELS)
# PROJECT (/e/project1/e-dev-2026d09-262) holds only pretrained models and datasets.
mkdir -p "${MAGI1_OUTPUTS}" "${MAGI1_LOGS}"
