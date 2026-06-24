#!/usr/bin/env bash
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

CUDA_VISIBLE_DEVICES=0 HYDRA_CONFIG=./configs/polygraph_eval_xsum.yaml /usr/bin/time -v python run_polygraph.py \
    ignore_exceptions=False \
    use_density_based_ue=False \
    batch_size=1 \
    subsample_train_dataset=100 \
    subsample_background_train_dataset=100 \
    subsample_eval_dataset=100 \
    model.path=meta-llama/Llama-3.1-8B \
    +model.attn_implementation=eager \
    cache_path=./workdir/paper/runtime \
    +generation_params.samples_n=5 \
    +train_pi=False \
    use_seq_ue=True \
    +run_pi_baselines=False \
    +run_baselines=False \
    +run_supervised_baselines=False \
    +target_train_metric=AlignScoreInv \
    +n_steps='[]' \
    +run_all_versions=False
wait
