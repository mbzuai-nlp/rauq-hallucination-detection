#!/usr/bin/env bash
set -e
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

run_eval truthfullqa meta-llama/Llama-3.1-8B ./workdir/paper/error_analysis AlignScore \
    +run_baselines=False \
    +run_supervised_baselines=False \
    +run_all_versions=False \
    +run_rauq_grid=False \
    +save_eval=True \
    +save_eval_full=True
