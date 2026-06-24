#!/usr/bin/env bash
set -e
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

run_paper_datasets meta-llama/Llama-3.1-8B ./workdir/paper/main_table +run_baselines=True +run_supervised_baselines=False
run_paper_datasets Qwen/Qwen2.5-7B ./workdir/paper/main_table +run_baselines=True +run_supervised_baselines=False
run_paper_datasets google/gemma-2-9b ./workdir/paper/main_table +run_baselines=True +run_supervised_baselines=False +model.use_cache=False
run_paper_datasets tiiuae/Falcon3-10B-Base ./workdir/paper/main_table +run_baselines=True +run_supervised_baselines=False
