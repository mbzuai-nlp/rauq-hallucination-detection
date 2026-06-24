#!/usr/bin/env bash
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

run_eval() {
    local dataset="$1"
    local model="$2"
    local cache_path="$3"
    local target_metric="$4"
    shift 4
    local eval_dataset="${dataset%_instruct}"
    local eval_size=2000

    case "${eval_dataset}" in
        truthfullqa)
            eval_size=817
            ;;
        sciq|medquad)
            eval_size=1000
            ;;
        gsm8k)
            eval_size=1319
            ;;
        samsum)
            eval_size=819
            ;;
        *)
            eval_size=2000
            ;;
    esac

    CUDA_VISIBLE_DEVICES=0 HYDRA_CONFIG="./configs/polygraph_eval_${dataset}.yaml" python run_polygraph.py \
        ignore_exceptions=False \
        use_density_based_ue=False \
        batch_size=1 \
        subsample_train_dataset=1000 \
        subsample_background_train_dataset=1000 \
        subsample_eval_dataset="${eval_size}" \
        model.path="${model}" \
        +model.attn_implementation=eager \
        cache_path="${cache_path}" \
        +generation_params.samples_n=5 \
        +train_pi=False \
        use_seq_ue=True \
        +run_pi_baselines=False \
        +target_train_metric="${target_metric}" \
        +n_steps='[]' \
        "$@"
    wait
}

run_paper_datasets() {
    local model="$1"
    local cache_path="$2"
    shift 2

    run_eval truthfullqa "${model}" "${cache_path}" AlignScore "$@"
    run_eval sciq "${model}" "${cache_path}" AlignScore "$@"
    run_eval mmlu "${model}" "${cache_path}" Accuracy "$@"
    run_eval triviaqa "${model}" "${cache_path}" AlignScore "$@"
    run_eval coqa "${model}" "${cache_path}" AlignScore "$@"
    run_eval samsum "${model}" "${cache_path}" AlignScoreInv "$@"
    run_eval cnn "${model}" "${cache_path}" AlignScoreInv "$@"
    run_eval wmt19_deen "${model}" "${cache_path}" Comet +metric_thr=0.85 "$@"
    run_eval wmt14_fren "${model}" "${cache_path}" Comet +metric_thr=0.85 "$@"
    run_eval gsm8k "${model}" "${cache_path}" Accuracy "$@"
    run_eval medquad "${model}" "${cache_path}" AlignScore "$@"
    run_eval xsum "${model}" "${cache_path}" AlignScoreInv "$@"
}

run_paper_instruct_datasets() {
    local model="$1"
    local cache_path="$2"
    shift 2

    run_eval truthfullqa_instruct "${model}" "${cache_path}" AlignScore "$@"
    run_eval sciq_instruct "${model}" "${cache_path}" AlignScore "$@"
    run_eval mmlu_instruct "${model}" "${cache_path}" Accuracy "$@"
    run_eval triviaqa_instruct "${model}" "${cache_path}" AlignScore "$@"
    run_eval coqa_instruct "${model}" "${cache_path}" AlignScore "$@"
    run_eval samsum_instruct "${model}" "${cache_path}" AlignScoreInv "$@"
    run_eval cnn_instruct "${model}" "${cache_path}" AlignScoreInv "$@"
    run_eval wmt19_deen_instruct "${model}" "${cache_path}" Comet +metric_thr=0.85 "$@"
    run_eval wmt14_fren_instruct "${model}" "${cache_path}" Comet +metric_thr=0.85 "$@"
    run_eval gsm8k_instruct "${model}" "${cache_path}" Accuracy "$@"
    run_eval medquad_instruct "${model}" "${cache_path}" AlignScore "$@"
    run_eval xsum_instruct "${model}" "${cache_path}" AlignScoreInv "$@"
}
