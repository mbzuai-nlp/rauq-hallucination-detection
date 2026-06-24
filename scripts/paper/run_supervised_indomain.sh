#!/usr/bin/env bash
set -e
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

run_paper_datasets meta-llama/Llama-3.1-8B ./workdir/paper/supervised_indomain \
    +run_baselines=False \
    +run_supervised_baselines=True \
    +run_all_versions=False
