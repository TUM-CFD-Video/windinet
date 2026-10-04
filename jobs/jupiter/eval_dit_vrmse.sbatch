#!/bin/bash
#SBATCH --job-name=wh_eval_dit_vrmse
#SBATCH --account=e-dev-2026d09-262
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00  # 500 test sims took 1.5h on one sng_pvc tile (job 546059)
#SBATCH --output=logs/jupiter/%x_%j.log
#SBATCH --error=logs/jupiter/%x_%j.err

# jupiter sibling of jobs/sng_pvc/eval_dit_vrmse.sbatch: launcher for
# scripts/eval_dit_vrmse.py -- VAE-only reconstruction vs full VAE + DiT
# rollout, pixel-space VRMSE (pooled / channel-mean / per channel / per gamma)
# plus latent VRMSE, per-sim values kept. Forward-only, plain python, 1 GPU.
#
# Defaults reproduce the thesis protocol (THESIS_RESULTS.md section 3, jobs
# 546059-546065): the standalone 256x256 test.h5, all 500 sims, 20 sampler
# steps (configs/dit/inference_dit.yaml), frame 0 encoded with the posterior
# mean, 3 gamma-spread npz samples for figures.
#
# Arguments:
#   PREPROCESSED_ROOT  dit_preprocessed/<name> dir the DiT was trained on
#                      (normalization.json + split_manifest.json)
#   DIT_CHECKPOINT     a model_weights_step_*.safetensors file, or a
#                      checkpoints/ dir (-> its newest model_weights_step_*,
#                      resolved when the job starts, so it can be chained on
#                      a training job)
#   VAE_CHECKPOINT     the vae_shockwave_best.safetensors those latents were
#                      encoded with; its run dir's training_config.yaml gives
#                      the pixel normalization
#   NUM_SAMPLES        optional, default 500
# Env: TEST_H5 (default the jupiter 256x256 test.h5; "none" = the manifest's
# val split instead), SAVE_VIS_SAMPLES (default 3), SAVE_NPZ_SAMPLES (default 3),
# NUM_INFERENCE_STEPS (default: config's 20).
#
# Results: logs/jupiter/eval_dit_vrmse/<job>/vrmse_summary.json (tracked);
# panels there are gitignored, npz figure fields go to scratch.
#
# Submit from the repo root, windinet env active:
#   sbatch jobs/jupiter/eval_dit_vrmse.sbatch <PREPROCESSED_ROOT> <DIT_CHECKPOINT> <VAE_CHECKPOINT> [NUM_SAMPLES]

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
if [[ -f sc_venv_template/activate.sh ]]; then
    set +eu  # modules.sh has a failing "module load mpi4py" and unset vars
    source sc_venv_template/activate.sh
    set -eu
fi
PYTHON=python

export SCRATCH="${SCRATCH_e_dev_2026d09_262}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WINDINET_HF_CACHE=/e/project1/e-dev-2026d09-262/wh_work/ltx_pretrained_wh
export HF_HUB_CACHE="${WINDINET_HF_CACHE}"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"  # windinet is not pip-installed in the venv
export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE

JOB_START_EPOCH=$(date +%s)
trap 'echo "end_time=$(date -Is) elapsed=$(( $(date +%s) - JOB_START_EPOCH ))s"' EXIT
trap 'exit 143' TERM
echo "job=${SLURM_JOB_ID} node=$(hostname) gpus=${CUDA_VISIBLE_DEVICES:-}"
echo "start_time=$(date -Is)"

USAGE="usage: sbatch jobs/jupiter/eval_dit_vrmse.sbatch <PREPROCESSED_ROOT> <DIT_CHECKPOINT> <VAE_CHECKPOINT> [NUM_SAMPLES]"
PREPROCESSED_ROOT=${1:?${USAGE}}
DIT_CHECKPOINT=${2:?${USAGE}}
VAE_CHECKPOINT=${3:?${USAGE}}
NUM_SAMPLES=${4:-500}
TEST_H5=${TEST_H5:-/e/project1/e-dev-2026d09-262/datasets/euler_mq_dataset/256x256_ds/test.h5}
SAVE_VIS_SAMPLES=${SAVE_VIS_SAMPLES:-3}
SAVE_NPZ_SAMPLES=${SAVE_NPZ_SAMPLES:-3}
NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS:-}

if [[ -d "${DIT_CHECKPOINT}" ]]; then
    DIT_CHECKPOINT=$(ls -1 "${DIT_CHECKPOINT}"/model_weights_step_*.safetensors 2>/dev/null | sort -V | tail -1 || true)
    [[ -n "${DIT_CHECKPOINT}" ]] || { echo "error: no model_weights_step_*.safetensors in $2" >&2; exit 1; }
fi
# Sibling ScalarEmbedding file of the same step (dit_trainer.py's save convention).
SCALAR_CHECKPOINT=${DIT_CHECKPOINT/model_weights_/scalar_embedding_}
NORMALIZATION=$(dirname "$(dirname "${VAE_CHECKPOINT}")")/training_config.yaml
OUT_DIR="logs/jupiter/eval_dit_vrmse/${SLURM_JOB_ID}"
RUN=$(basename "$(dirname "$(dirname "${DIT_CHECKPOINT}")")")
NPZ_DIR="${SCRATCH}/wh_work/figure_data/dit/${RUN}"
for f in "${PREPROCESSED_ROOT}/normalization.json" "${DIT_CHECKPOINT}" "${SCALAR_CHECKPOINT}" "${VAE_CHECKPOINT}" "${NORMALIZATION}"; do
    [[ -e "${f}" ]] || { echo "error: missing ${f}" >&2; exit 1; }
done

EXTRA_ARGS=()
[[ "${TEST_H5}" != none ]] && EXTRA_ARGS+=(--test_h5 "${TEST_H5}")
[[ -n "${NUM_INFERENCE_STEPS}" ]] && EXTRA_ARGS+=(--num_inference_steps "${NUM_INFERENCE_STEPS}")
[[ "${SAVE_NPZ_SAMPLES}" != 0 ]] && EXTRA_ARGS+=(--save_npz_samples "${SAVE_NPZ_SAMPLES}" --npz_dir "${NPZ_DIR}")

echo "PREPROCESSED_ROOT=${PREPROCESSED_ROOT}"
echo "DIT_CHECKPOINT=${DIT_CHECKPOINT}"
echo "SCALAR_CHECKPOINT=${SCALAR_CHECKPOINT}"
echo "VAE_CHECKPOINT=${VAE_CHECKPOINT} NORMALIZATION=${NORMALIZATION}"
echo "TEST_H5=${TEST_H5} NUM_SAMPLES=${NUM_SAMPLES} NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS:-<config default>} OUT_DIR=${OUT_DIR}"
nvidia-smi --query-gpu=name,memory.total --format=csv || true
mkdir -p "${OUT_DIR}"
printf '%s\t%s\t%s\t%s\t%s\n' "$(date -Is)" "${SLURM_JOB_ID}" "${SLURM_JOB_NAME}" \
    "${DIT_CHECKPOINT}" "${OUT_DIR}" >> logs/jupiter/INDEX.tsv

"${PYTHON}" scripts/eval_dit_vrmse.py configs/dit/inference_dit.yaml \
    --preprocessed_data_root "${PREPROCESSED_ROOT}" \
    --checkpoint "${DIT_CHECKPOINT}" \
    --scalar_checkpoint "${SCALAR_CHECKPOINT}" \
    --vae_checkpoint "${VAE_CHECKPOINT}" \
    --normalization "${NORMALIZATION}" \
    --num_samples "${NUM_SAMPLES}" \
    --save_vis_samples "${SAVE_VIS_SAMPLES}" \
    --out_dir "${OUT_DIR}" \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
