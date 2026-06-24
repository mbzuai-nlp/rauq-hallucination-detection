#!/usr/bin/env bash
set -e
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

run_paper_datasets meta-llama/Llama-3.1-8B ./workdir/paper/layer_subset \
    +run_baselines=False \
    +run_supervised_baselines=False \
    +run_all_versions=False \
    +run_rauq_grid=False \
    +run_layer_ablation=True
