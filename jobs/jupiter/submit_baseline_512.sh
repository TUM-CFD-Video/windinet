#!/bin/bash
# Jupiter baseline at 512x512, end to end (VAE -> vaetest | encode -> DiT ->
# diteval): finetune_vae_jupiter_baseline_512res + train_dit_jupiter_baseline_512res,
# otherwise identical to the 256x256 baseline. Every stage reads the resolution
# from the VAE run (train.h5 / test.h5 under 512x512_orig).
#
# 512x512 has 4x the pixels and 4x the DiT tokens, and nothing is measured at
# this resolution on jupiter yet, so stages get the 12h booster maximum plus
# resume jobs (surplus resumes exit within a minute; see submit_pipeline.sh):
#   VAE     BATCH_SIZE=1 (accum 8, effective batch 32), 12h + 2 resumes
#           (256x256: ~3.5h at batch 4 -> expect ~14h+)
#   encode  12h + 1 resume (skips latents already written)
#   DiT     12h + 3 resumes (256x256: ~6.7h for 8k steps)
#   evals   VAE test 4h, VAE + DiT test 12h (no resume: rerun if it times out)
# Any of these can be overridden from the environment (see submit_pipeline.sh).
#
# Run from the repo root on a jupiter login node, windinet env active, once
# euler_mq_dataset/jupiter/download_from_hf.py 512x512_orig has finished:
#   bash jobs/jupiter/submit_baseline_512.sh
set -euo pipefail

DATA=/e/project1/e-dev-2026d09-262/datasets/euler_mq_dataset/512x512_orig
n_train=$(ls "${DATA}"/train_subsets/train_gamma*.hdf5 2>/dev/null | wc -l)
n_test=$(ls "${DATA}"/test_subsets/test_gamma*.hdf5 2>/dev/null | wc -l)
if [ ! -e "${DATA}/train.h5" ] || [ ! -e "${DATA}/test.h5" ] || [ "$n_train" -ne 10 ] || [ "$n_test" -ne 10 ]; then
    echo "error: 512x512 dataset incomplete under ${DATA} (train/test.h5 + ${n_train}/10 train, ${n_test}/10 test shards)" >&2
    echo "       finish: python euler_mq_dataset/jupiter/download_from_hf.py 512x512_orig" >&2
    exit 1
fi

export BATCH_SIZE=${BATCH_SIZE:-1}
export VAE_TIME=${VAE_TIME:-12:00:00} VAE_RESUMES=${VAE_RESUMES:-2}
export VAETEST_TIME=${VAETEST_TIME:-04:00:00}
export ENCODE_TIME=${ENCODE_TIME:-12:00:00} ENCODE_RESUMES=${ENCODE_RESUMES:-1}
export DIT_TIME=${DIT_TIME:-12:00:00} DIT_RESUMES=${DIT_RESUMES:-3}
export EVAL_TIME=${EVAL_TIME:-12:00:00}
exec bash jobs/jupiter/submit_pipeline.sh \
    configs/finetune_vae/finetune_vae_jupiter_baseline_512res.yaml \
    configs/dit/train_dit_jupiter_baseline_512res.yaml
