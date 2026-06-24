#!/usr/bin/env bash
set -e
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

run_paper_instruct_datasets meta-llama/Llama-3.1-8B-Instruct ./workdir/paper/instruct +run_baselines=True +run_supervised_baselines=False
run_paper_instruct_datasets openai/gpt-oss-20b ./workdir/paper/instruct \
    +run_baselines=True \
    +run_supervised_baselines=False \
    +run_all_versions=False \
    +model.dtype=bfloat16 \
    +max_new_tokens=256
