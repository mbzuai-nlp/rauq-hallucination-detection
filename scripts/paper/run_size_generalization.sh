#!/usr/bin/env bash
set -e
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

run_paper_datasets HuggingFaceTB/SmolLM2-360M ./workdir/paper/size_generalization +run_baselines=True +run_supervised_baselines=False +run_all_versions=False
run_paper_datasets unsloth/Llama-3.2-1B ./workdir/paper/size_generalization +run_baselines=True +run_supervised_baselines=False +run_all_versions=False
run_paper_datasets meta-llama/Llama-3.1-70B ./workdir/paper/size_generalization \
    +run_baselines=True \
    +run_supervised_baselines=False \
    +run_all_versions=False \
    +model.uniform_memory=True \
    +model.dtype=bfloat16 \
    deberta_batch_size=1 \
    +max_input_tokens=1024
