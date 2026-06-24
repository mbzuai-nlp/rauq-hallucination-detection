#!/usr/bin/env bash
set -e
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

run_eval wmt14_fren meta-llama/Llama-3.1-8B ./workdir/paper/single_head Comet \
    +metric_thr=0.85 \
    +run_baselines=False \
    +run_supervised_baselines=False \
    +run_all_versions=False \
    +run_rauq_grid=True \
    +run_single_head=True

run_eval coqa meta-llama/Llama-3.1-8B ./workdir/paper/single_head AlignScore \
    +aggregation_func=all \
    +run_baselines=False \
    +run_supervised_baselines=False \
    +run_all_versions=False \
    +run_rauq_grid=True \
    +run_single_head=True
