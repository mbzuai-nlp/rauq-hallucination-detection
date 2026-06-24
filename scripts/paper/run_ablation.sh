#!/usr/bin/env bash
set -e
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

run_paper_datasets meta-llama/Llama-3.1-8B ./workdir/paper/ablation \
    +run_baselines=False \
    +run_supervised_baselines=False \
    +run_all_versions=False \
    +run_rauq_grid=False \
    +run_alpha_ablation=True \
    +run_token_aggregation_ablation=True \
    +run_layer_aggregation_ablation=True \
    +run_head_ablation=True \
    +run_formula_ablation=True \
    +run_delta_ablation=True
