# Efficient Hallucination Detection for LLMs Using Uncertainty-Aware Attention Heads

Large language models are highly capable, but they still produce fluent answers that can be factually wrong or unsupported by the input. This repository accompanies the paper "Efficient Hallucination Detection for LLMs Using Uncertainty-Aware Attention Heads" and provides code for **Recurrent Attention-Based Uncertainty Quantification (RAUQ)**, an unsupervised method for hallucination detection in white-box LLMs. RAUQ is based on the observation that, when a model generates incorrect information, specific uncertainty-aware attention heads reduce their focus on preceding tokens. The method automatically identifies these heads, combines their attention patterns with token-level confidence in a recurrent scheme, and produces a sequence-level uncertainty score in a single forward pass. Across twelve datasets, three generation tasks, and nine LLMs, RAUQ consistently outperforms state-of-the-art UQ baselines while adding less than 1% computational overhead, making it a lightweight plug-and-play method for real-time hallucination detection.

The main entry point is `run_polygraph.py`. Final paper commands live in `scripts/paper/`.

**Note:** This repository is intended only for reproducing the experiments from the paper. The current and recommended RAUQ implementation is maintained in [IINemo/lm-polygraph](https://github.com/IINemo/lm-polygraph).

## Repository layout

```text
configs/          Hydra configs for the final paper datasets
scripts/paper/    Clean launch scripts mapped to paper tables
notebooks/        Result tables and attention/head analysis notebooks
run_polygraph.py  Main experiment runner
rauq.py           Local RAUQ implementation used by the runner
requirements.txt  Python dependencies, including lm-polygraph
```

Generated outputs are written under `workdir/` and should not be committed.

## Method implementation

The main method is implemented in `rauq.py` as the `RAUQ` class. It is an `lm-polygraph` sequence-level estimator that consumes generation attentions (`attention_all`) and token log-likelihoods (`greedy_log_likelihoods`); the entropy variant additionally uses token entropy, and the single-head variant uses training attentions to select one head.

RAUQ extracts attention to the previous token from selected layers and heads, combines it recurrently with token-level confidence using `alpha`, then aggregates token scores (`token_aggregation`) and layer scores (`aggregation`) into one uncertainty score per generated sequence. The method is added to the experiment pipeline in `run_polygraph.py`, where the paper setting and ablations vary `alpha`, entropy usage, layer/head selection, token aggregation, and layer aggregation.

## Installation

```bash
git clone https://github.com/mbzuai-nlp/rauq-hallucination-detection.git
cd rauq-hallucination-detection
pip install -r requirements.txt
```

Some experiments require gated model access on Hugging Face, for example Llama models. The 70B and GPT-OSS runs are expensive and need appropriate GPU memory.

## Paper commands

Run scripts from the repository root.

| Paper result | Command |
|---|---|
| Main base-model PRR results, detailed model tables | `bash scripts/paper/run_main_table.sh` |
| Attention Score modifications | `bash scripts/paper/run_attention_score_modifications.sh` |
| RAUQ ablations | `bash scripts/paper/run_ablation.sh` |
| Layer subset analysis | `bash scripts/paper/run_layer_subset.sh` |
| Dynamic head vs single head | `bash scripts/paper/run_single_head.sh` |
| LLM Size generalization | `bash scripts/paper/run_size_generalization.sh` |
| Instruction-tuned models | `bash scripts/paper/run_instruct.sh` |
| Supervised baselines, in-domain | `bash scripts/paper/run_supervised_indomain.sh` |
| Supervised baselines, out-of-domain | `bash scripts/paper/run_supervised_ood.sh` |
| RAUQ with LIG | `bash scripts/paper/run_lig.sh` |

Tables derived from existing outputs, such as ROC-AUC summaries and dataset/generation statistics, are computed from the saved `workdir/paper/` runs and the configs in `configs/`.

## Example

```bash
bash scripts/paper/run_main_table.sh
```

The script runs the final base models on the final paper datasets and saves outputs under:

```text
workdir/paper/main_table/
```

To run a narrower experiment, edit the corresponding script in `scripts/paper/`. Fixed settings such as dataset sizes, batch size, and output directories are kept in the scripts for reproducibility.

## Analysis notebooks

- `notebooks/results_table_from_paths.ipynb`: build result tables from saved output paths.
- `notebooks/attention_first_example.ipynb`: reproduce the first attention visualization example.

## Citation

@inproceedings{vazhentsev2026efficient,
  title = {Efficient Hallucination Detection for {LLM}s Using Uncertainty-Aware Attention Heads},
  author = {Vazhentsev, Artem and Rvanova, Lyudmila and Kuzmin, Gleb and Fadeeva, Ekaterina and Lazichny, Ivan and Panchenko, Alexander and Panov, Maxim and Sachan, Mrinmaya and Nakov, Preslav and Baldwin, Timothy and Shelmanov, Artem},
  booktitle = {Forty-third International Conference on Machine Learning},
  year = {2026},
  address = {Seoul, South Korea},
  url = {https://arxiv.org/abs/2505.20045},
}
