#!/usr/bin/env python3
import codecs
import re
import hydra
import importlib
import itertools
import os
import sys
import torch
import transformers
from pathlib import Path
from typing import Dict

import logging
from dataclasses import asdict
from omegaconf import OmegaConf

log = logging.getLogger()
sys.path.append(str(Path(__file__).resolve().parent / "utils"))

from lm_polygraph.utils.manager import UEManager
from utils.dataset import Dataset
from lm_polygraph.utils.model import WhiteboxModel, create_ensemble
from lm_polygraph.utils.processor import Logger
from lm_polygraph.generation_metrics.accuracy import AccuracyMetric
from lm_polygraph.generation_metrics.rouge import RougeMetric
from lm_polygraph.generation_metrics.aggregated_metric import AggregatedMetric
from lm_polygraph.generation_metrics.alignscore import AlignScore
from lm_polygraph.generation_metrics.comet import Comet
from lm_polygraph.estimators import *
from lm_polygraph.estimators.ensemble_token_measures import all_token_estimators
from lm_polygraph.estimators.ensemble_sequence_measures import (
    all_ep_estimators,
    all_pe_estimators,
)
from lm_polygraph.estimators.ensemble_token_measures import *
from lm_polygraph.ue_metrics import *
from lm_polygraph.utils.generation_parameters import (
    GenerationParametersFactory,
)
from lm_polygraph.utils.builder_enviroment_stat_calculator import (
    BuilderEnvironmentStatCalculator,
)
from lm_polygraph.utils.factory_stat_calculator import StatCalculatorContainer

from unsupervised_baselines.simple_focus import SimpleFocus
from unsupervised_baselines.grads_methods import IntegratedGradients
from unsupervised_baselines.llm_check_attention import LLMCheckAttention

from utils.register_default_stat_calculators import (
    register_default_stat_calculators,
)
from utils.tokenizer_decode_patch import apply_patch
from utils.gpt_oss_full_patch import apply_patch as apply_gpt_oss_full_patch

from rauq import RAUQ


def _remove_gpt_oss_control_tokens(text: str) -> str:
    """
    Removes GPT OSS control tokens from generated text.
    These tokens appear when using the reasoning mode workaround.
    """
    # Remove <|channel|>...<|message|> prefix patterns
    text = re.sub(r"<\|channel\|>analysis<\|message\|>", "", text)
    text = re.sub(r"<\|channel\|>final<\|message\|>", "", text)
    text = re.sub(r"<\|channel\|>commentary<\|message\|>", "", text)

    # Remove other GPT OSS special tokens that might appear
    text = re.sub(r"<\|start\|>assistant$", "", text)  # at end only
    text = re.sub(r"<\|end\|>$", "", text)  # at end only
    text = re.sub(r"<\|return\|>$", "", text)  # at end only

    # Clean up any leading/trailing whitespace
    text = text.strip()

    return text


def _batch_max_new_tokens(batch_max_new_tokens, default):
    if batch_max_new_tokens is None:
        return default
    if isinstance(batch_max_new_tokens, (list, tuple)):
        if len(batch_max_new_tokens) == 0:
            return default
        first_value = batch_max_new_tokens[0]
        if any(value != first_value for value in batch_max_new_tokens):
            raise ValueError(
                "Batch contains mixed max_new_tokens values. "
                "Use Dataset.iter_with_metadata() to split it first."
            )
        return first_value
    return batch_max_new_tokens


class DatasetAwareUEManager(UEManager):
    def _process(self, iterable_data, batch_callback):
        iterable = (
            iterable_data.iter_with_metadata()
            if hasattr(iterable_data, "iter_with_metadata")
            else iterable_data
        )
        for batch_i, batch in enumerate(iterable):
            batch_max_new_tokens = None
            raw_inp_texts = None
            if len(batch) == 4:
                inp_texts, raw_inp_texts, target_texts, batch_max_new_tokens = batch
                images = None
            elif len(batch) == 3:
                inp_texts, target_texts, images = batch
            elif len(batch) == 2:
                inp_texts, target_texts = batch
                images = None
            else:
                raise ValueError(
                    f"Expected batch with 2, 3 or 4 elements, got {len(batch)}"
                )

            batch_stats: Dict[str, torch.Tensor] = {}
            for key, val in [
                ("input_texts", inp_texts),
                ("target_texts", target_texts),
            ]:
                self.stats[key] += val
                batch_stats[key] = val
            if raw_inp_texts is not None:
                batch_stats["raw_input_texts"] = raw_inp_texts
                if "raw_input_texts" in self.stats:
                    self.stats["raw_input_texts"] += raw_inp_texts

            if images is not None and not (
                isinstance(images, list) and all(img is None for img in images)
            ):
                from lm_polygraph.utils.dataset import Dataset as PolygraphDataset

                self.stats["images"] += PolygraphDataset.get_images(images)
                batch_stats["images"] = PolygraphDataset.get_images(images)

            batch_stats["model"] = self.model
            batch_stats["layers"] = self.layers

            old_max_new_tokens = self.max_new_tokens
            self.max_new_tokens = _batch_max_new_tokens(
                batch_max_new_tokens, old_max_new_tokens
            )
            try:
                batch_stats = self.calculate(
                    batch_stats, self.stat_calculators, inp_texts
                )
            finally:
                self.max_new_tokens = old_max_new_tokens

            batch_estimations, bad_estimators = self.estimate(
                batch_stats, self.estimators
            )

            batch_callback(
                batch_i, target_texts, batch_stats, batch_estimations, bad_estimators
            )
            torch.cuda.empty_cache()
            import gc

            gc.collect()

        return self.estimations


def _decode_config_text(value):
    return codecs.decode(value, "unicode_escape") if isinstance(value, str) else value


def _normalize_multiref_targets(dataset, multiref):
    if len(dataset.y) == 0:
        return
    if multiref:
        if not isinstance(dataset.y[0], list):
            dataset.y = [[y] for y in dataset.y]
    elif isinstance(dataset.y[0], list):
        dataset.y = [y[0] for y in dataset.y]


def _load_ood_train_dataset(args, seed, cache_kwargs):
    train_dataset = None
    k_ds = 1
    while getattr(args, f"train_dataset_{k_ds}", False):
        train_dataset_k = Dataset.load(
            getattr(args, f"train_dataset_{k_ds}"),
            getattr(args, f"train_text_column_{k_ds}"),
            getattr(args, f"train_label_column_{k_ds}"),
            batch_size=args.batch_size,
            prompt=_decode_config_text(getattr(args, f"train_prompt_{k_ds}")),
            description=_decode_config_text(
                getattr(args, f"train_description_{k_ds}", "")
            ),
            few_shot_prompt=_decode_config_text(
                getattr(args, f"train_few_shot_prompt_{k_ds}", "")
            ),
            mmlu_max_subject_size=getattr(args, "mmlu_max_subject_size", 100),
            n_shot=getattr(args, f"train_n_shot_{k_ds}", 5),
            few_shot_split=getattr(args, f"few_shot_split_{k_ds}", "train"),
            split=getattr(args, f"train_split_{k_ds}", "train"),
            max_new_tokens=getattr(args, f"max_new_tokens_{k_ds}", 100),
            size=10_000,
            load_from_disk=args.load_from_disk,
            **cache_kwargs,
        )

        if args.subsample_train_dataset != -1:
            train_dataset_k.subsample(args.subsample_train_dataset, seed=seed)

        _normalize_multiref_targets(train_dataset_k, getattr(args, "multiref", False))

        if train_dataset is None:
            train_dataset = train_dataset_k
        else:
            train_dataset.concat(
                train_dataset_k.x,
                train_dataset_k.raw_x,
                train_dataset_k.y,
                train_dataset_k.max_new_tokens,
            )
        k_ds += 1

    return train_dataset


hydra_config = Path(os.environ["HYDRA_CONFIG"])


@hydra.main(
    version_base=None,
    config_path=str(hydra_config.parent),
    config_name=str(hydra_config.name),
)
def main(args):
    save_path = os.getcwd()
    log.info(f"Main directory: {save_path}")
    os.chdir(hydra.utils.get_original_cwd())

    save_path = args.save_path if "save_path" in args else save_path

    if args.seed is None or len(args.seed) == 0:
        args.seed = [1]

    model_kwargs = get_model_kwargs(args)

    cache_kwargs = {}
    if os.environ.get("HF_DATASETS_OFFLINE", "").strip() == "1":
        cache_kwargs = {"cache_dir": args.cache_path}

    for seed in args.seed:
        log.info("=" * 100)
        log.info(f"SEED: {seed}")

        log.info(f"Loading model {args.model.path}...")
        transformers.set_seed(seed)

        if "gpt-oss" in args.model.path.lower():
            from transformers import AutoModelForCausalLM, AutoTokenizer

            model_path = args.model.path
            model_type = "CausalLM"
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                device_map=args.model.device_map,
                attn_implementation="eager",
                dtype="bfloat16",
            )
            model.eval()

            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                padding_side="left",
                add_bos_token=getattr(args.model, "add_bos_token", True),
                enable_thinking=False,
            )
            print(
                "\n\nenable_thinking=False is set for the tokenizer, which will disable the special thinking token."
            )

            # Workaround to disable reasoning mode for GPT OSS models by patching chat template
            if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
                original_template = tokenizer.chat_template
                # Replace the assistant generation prompt to bypass reasoning mode
                patched_template = original_template.replace(
                    "{%- if add_generation_prompt -%}\n<|start|>assistant",
                    "{%- if add_generation_prompt -%}\n<|start|>assistant<|channel|>analysis<|message|><|end|><|start|>assistant",
                )
                if patched_template != original_template:
                    tokenizer.chat_template = patched_template
                    print(
                        "Applied chat_template patch to disable reasoning mode for GPT OSS model."
                    )

            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            generation_params = GenerationParametersFactory.from_params(
                yaml_config=getattr(args, "generation_params", {}),
                native_config=asdict(model.config),
            )

            model = WhiteboxModel(
                model,
                tokenizer,
                model_path,
                model_type,
                generation_params,
                instruct=getattr(args, "instruct", False),
            )
            apply_patch(_remove_gpt_oss_control_tokens, model)
            max_new_tokens = getattr(args, "max_new_tokens", 100)
            apply_gpt_oss_full_patch(model, max_new_tokens=max_new_tokens)

        else:
            model = WhiteboxModel.from_pretrained(
                args.model.path,
                getattr(args, "generation_params", {}),
                device_map=args.model.device_map,
                add_bos_token=getattr(args.model, "add_bos_token", True),
                **cache_kwargs,
                **model_kwargs,
            )
            model.instruct = getattr(args, "instruct", False)
        if getattr(args, "reasoning", False):
            model.tokenizer.eos_think_id = model.tokenizer("</think>")["input_ids"][0]
        if args.model.ensemble:
            # Only MC-ensembles for now
            log.info(f"Creating ensemble...")
            ensemble_model = create_ensemble(
                model_paths=[args.model.path],
                mc=True,
                seed=args.seed[0],
                ensembling_mode=args.model.ensembling_mode,
                mc_seeds=args.model.mc_seeds,
                dropout_rate=float(args.model.dropout_rate),
                **cache_kwargs,
                **model_kwargs,
            )
        else:
            ensemble_model = None

        log.info("Done with loading model.")

        log.info(f"Loading dataset {args.dataset}...")
        dataset = Dataset.load(
            args.dataset,
            args.text_column,
            args.label_column,
            batch_size=args.batch_size,
            prompt=args.prompt,
            description=getattr(args, "description", ""),
            few_shot_prompt=getattr(args, "few_shot_prompt", ""),
            mmlu_max_subject_size=getattr(args, "mmlu_max_subject_size", 100),
            n_shot=getattr(args, "n_shot", 5),
            few_shot_split=getattr(args, "few_shot_split", "train"),
            split=args.eval_split,
            load_from_disk=args.load_from_disk,
            max_new_tokens=getattr(args, f"max_new_tokens", 100),
            **cache_kwargs,
        )

        estimators = []
        estimators += get_ue_methods(args, model)
        density_based_ue_methods = get_density_based_ue_methods(args, model.model_type)
        estimators += density_based_ue_methods

        train_dataset = None
        background_train_dataset = None
        if any([not getattr(method, "is_fitted", True) for method in estimators]) and (
            not getattr(args, "kfolds", False)
        ):
            if getattr(args, "is_ood_exps", False) and getattr(
                args, "train_dataset_1", False
            ):
                train_dataset = _load_ood_train_dataset(args, seed, cache_kwargs)
            elif (args.train_dataset is not None) and (
                args.train_dataset != args.dataset
            ):
                train_dataset = Dataset.load(
                    args.train_dataset,
                    args.text_column,
                    args.label_column,
                    batch_size=args.batch_size,
                    prompt=args.prompt,
                    description=getattr(args, "description", ""),
                    few_shot_prompt=getattr(args, "few_shot_prompt", ""),
                    mmlu_max_subject_size=getattr(args, "mmlu_max_subject_size", 100),
                    n_shot=getattr(args, "n_shot", 5),
                    few_shot_split=getattr(args, "few_shot_split", "train"),
                    split=args.train_split,
                    size=10_000,
                    load_from_disk=args.load_from_disk,
                    max_new_tokens=getattr(args, f"max_new_tokens", 100),
                    **cache_kwargs,
                )
            elif args.train_test_split:
                (
                    X_train,
                    X_test,
                    X_raw_train,
                    X_raw_test,
                    y_train,
                    y_test,
                    max_new_tokens_train,
                    max_new_tokens_test,
                ) = dataset.train_test_split(
                    test_size=args.test_split_size, seed=seed, split=args.eval_split
                )
                train_dataset = Dataset(
                    x=X_train,
                    raw_x=X_raw_train,
                    y=y_train,
                    max_new_tokens=getattr(args, "max_new_tokens", 100),
                    batch_size=args.batch_size,
                )
            else:
                train_dataset = Dataset.load(
                    args.dataset,
                    args.text_column,
                    args.label_column,
                    batch_size=args.batch_size,
                    prompt=args.prompt,
                    description=getattr(args, "description", ""),
                    few_shot_prompt=getattr(args, "few_shot_prompt", ""),
                    mmlu_max_subject_size=getattr(args, "mmlu_max_subject_size", 100),
                    n_shot=getattr(args, "n_shot", 5),
                    few_shot_split=getattr(args, "few_shot_split", "train"),
                    split=args.train_split,
                    size=10_000,
                    load_from_disk=args.load_from_disk,
                    max_new_tokens=getattr(args, f"max_new_tokens", 100),
                    **cache_kwargs,
                )
            if args.subsample_train_dataset != -1 and not getattr(
                args, "is_ood_exps", False
            ):
                train_dataset.subsample(args.subsample_train_dataset, seed=seed)

        if any([not getattr(method, "is_fitted", False) for method in estimators]):
            try:
                background_train_dataset = Dataset.load(
                    args.background_train_dataset,
                    args.background_train_dataset_text_column,
                    args.background_train_dataset_label_column,
                    batch_size=args.batch_size,
                    data_files=args.background_train_dataset_data_files,
                    split="train",
                    size=100_000,
                    load_from_disk=args.background_load_from_disk,
                    **cache_kwargs,
                )
                if args.subsample_background_train_dataset != -1:
                    background_train_dataset.subsample(
                        args.subsample_background_train_dataset, seed=seed
                    )
            except:
                pass

        if args.subsample_eval_dataset != -1:
            dataset.subsample(args.subsample_eval_dataset, seed=seed)

        log.info("Done with loading data.")

        generation_metrics = get_generation_metrics(args)
        ue_metrics = get_ue_metrics(args)

        builder_env_stat_calc = BuilderEnvironmentStatCalculator(model=model)
        available_stat_calculators = get_stat_calculator_names(args)

        man = DatasetAwareUEManager(
            dataset,
            model,
            estimators,
            builder_env_stat_calc=builder_env_stat_calc,
            available_stat_calculators=available_stat_calculators,
            generation_metrics=generation_metrics,
            ue_metrics=ue_metrics,
            processors=[
                Logger(),
            ],
            ignore_exceptions=args.ignore_exceptions,
            max_new_tokens=args.max_new_tokens,
            save_stats=getattr(args, "save_stats", []),
            log_time=getattr(args, "log_time", False),
        )

        man()

        man.save(save_path + f"/ue_manager_seed{seed}")


def get_ue_metrics(args):
    ue_metrics = [
        PredictionRejectionArea(),
        PredictionRejectionArea(max_rejection=0.5),
    ]
    if getattr(args, "use_claim_ue", False) or getattr(args, "train_claim_pi", False):
        ue_metrics += [
            ROCAUC(),
            PRAUC(),
        ]
    return ue_metrics


def get_density_based_ue_methods(args, model_type):
    estimators = []
    if args.use_density_based_ue:
        if getattr(args, "parameters_path", False):
            parameters_path = args.parameters_path
        else:
            dataset_name = (
                args.dataset
                if isinstance(args.dataset, str)
                else "_".join(args.dataset)
            )
            dataset_name = dataset_name.split("/")[-1].split(".")[0]
            model_name = args.model.path.split("/")[-1]
            parameters_path = (
                f"{args.cache_path}/density_stats/{dataset_name}/{model_name}"
            )

        if model_type == "Seq2SeqLM":
            estimators += [
                MahalanobisDistanceSeq("encoder", parameters_path=parameters_path),
                MahalanobisDistanceSeq("decoder", parameters_path=parameters_path),
                RelativeMahalanobisDistanceSeq(
                    "encoder", parameters_path=parameters_path
                ),
                RelativeMahalanobisDistanceSeq(
                    "decoder", parameters_path=parameters_path
                ),
                RDESeq("encoder", parameters_path=parameters_path),
                RDESeq("decoder", parameters_path=parameters_path),
                PPLMDSeq("encoder", md_type="MD", parameters_path=parameters_path),
                PPLMDSeq("encoder", md_type="RMD", parameters_path=parameters_path),
                PPLMDSeq("decoder", md_type="MD", parameters_path=parameters_path),
                PPLMDSeq("decoder", md_type="RMD", parameters_path=parameters_path),
            ]
        else:
            estimators += [
                MahalanobisDistanceSeq("decoder", parameters_path=parameters_path),
                RelativeMahalanobisDistanceSeq(
                    "decoder", parameters_path=parameters_path
                ),
                RDESeq("decoder", parameters_path=parameters_path),
                PPLMDSeq("decoder", md_type="MD", parameters_path=parameters_path),
                PPLMDSeq("decoder", md_type="RMD", parameters_path=parameters_path),
            ]
    return estimators


def get_ue_methods(args, model):
    estimators = []

    metric_name = getattr(args, "target_train_metric", "AlignScore")
    use_accuracy = False
    use_alignscore = True
    use_comet = False
    if getattr(args, "is_ood_exps", False):
        use_alignscore = True
        metric_name = "AlignScoreMean"
        metric = AlignScore(batch_size=4, return_mean=True)
    elif metric_name == "Accuracy":
        use_accuracy = True
        use_alignscore = False
        metric = AccuracyMetric(
            target_ignore_regex=getattr(args, "target_ignore_regex", None),
            output_ignore_regex=getattr(args, "output_ignore_regex", None),
            normalize=getattr(args, "normalize", False),
        )
    elif metric_name == "Comet":
        use_comet = True
        use_alignscore = False
        metric = Comet(source_ignore_regex=getattr(args, "source_ignore_regex", None))
    elif metric_name == "AlignScore":
        use_alignscore = True
        metric = AlignScore(batch_size=4)
    elif metric_name == "AlignScoreMean":
        use_alignscore = True
        metric = AlignScore(batch_size=4, return_mean=True)
    elif metric_name == "AlignScoreInv":
        use_alignscore = True
        metric = AlignScore(batch_size=4, target_is_claims=False)

    aggregated = getattr(args, "multiref", False)
    if aggregated:
        metric = AggregatedMetric(base_metric=metric)

    dataset_name = (
        args.dataset if isinstance(args.dataset, str) else "_".join(args.dataset)
    )
    dataset_name = dataset_name.split("/")[-1].split(".")[0]
    model_name = args.model.path.split("/")[-1]
    parameters_path = f"{args.save_path}/tad_stats/"

    if "gpt-oss" in args.model.path.lower() and (
        getattr(args, "max_new_tokens", 100) > 128
    ):
        layer_types = model.model.config.layer_types
        full_attn_layer_ids = [
            i for i, t in enumerate(layer_types) if t != "sliding_attention"
        ]
        hidden_layers = list(range(len(full_attn_layer_ids)))[:-1] + [-1]
        mid_layer = int(len(full_attn_layer_ids) // 2)
    else:
        hidden_layers = list(range(model.model.config.num_hidden_layers))[:-1] + [-1]
        mid_layer = int(model.model.config.num_hidden_layers // 2)

    if args.use_seq_ue:
        estimators += [
            MaximumSequenceProbability(),
            Perplexity(),
            MeanTokenEntropy(),
            Focus(
                gamma=getattr(args, "focus_gamma", 0.9),
                p=getattr(args, "focus_p", 0.01),
                model_name=args.model.path,
                path=getattr(
                    args,
                    "focus_path",
                    f"{getattr(args, 'cache_path', args.save_path)}/focus/{args.model.path}/token_idf.pkl",
                ),
                idf_dataset=getattr(
                    args,
                    "focus_idf_dataset",
                    "LM-Polygraph/RedPajama-Data-100-Sample-For-Test",
                ),
                trust_remote_code=getattr(args, "focus_trust_remote_code", True),
                idf_seed=getattr(args, "focus_idf_seed", 42),
                idf_dataset_size=getattr(args, "focus_idf_dataset_size", -1),
                spacy_path=getattr(args, "focus_spacy_path", "en_core_web_sm"),
                idf_dataset_text_column=getattr(
                    args, "focus_idf_dataset_text_column", "text"
                ),
            ),
            SimpleFocus(reccurent=True),
            LLMCheckAttention(
                layer=mid_layer,
                aggregation="sum",
                parameters_path=parameters_path,
            ),
        ]

        if getattr(args, "run_attention_score_ablation", False):
            estimators += [
                LLMCheckAttention(layer=mid_layer, gen_only=True),
                LLMCheckAttention(layer=mid_layer, one_head=True),
                LLMCheckAttention(layer=mid_layer, gen_only=True, one_head=True),
            ]

        if getattr(args, "run_baselines", False):
            estimators += [
                MonteCarloSequenceEntropy(),
                MonteCarloNormalizedSequenceEntropy(),
                LexicalSimilarity(metric="rougeL"),
                NumSemSets(),
                EigValLaplacian(similarity_score="NLI_score", affinity="entail"),
                DegMat(similarity_score="NLI_score", affinity="entail"),
                Eccentricity(similarity_score="NLI_score", affinity="entail"),
                SemanticEntropy(),
                SAR(),
                TokenSAR(),
                ClaimConditionedProbability(),
                SentenceSAR(),
                EigenScore(),
                LUQ(),
                SemanticDensity(),
            ]
        if getattr(args, "run_lig", False):
            estimators += [
                IntegratedGradients(model, model_name, positive=True, reccurent=True),
            ]

        metric_thr = getattr(args, "metric_thr", 0.3)

        if getattr(args, "run_supervised_baselines", False):
            estimators += [
                SAPLMA(model_name=args.model.path),
                MIND(metric_thr=metric_thr),
                Sheeps(metric_thr=metric_thr, model_name=args.model.path),
                SATRMD(
                    metric_thr=metric_thr,
                    model_name=args.model.path,
                    base_method="RelativeTokenMahalanobis",
                ),
                LookBackLens(metric_thr=metric_thr),
                TAD(),
            ]

        if "gpt-oss" in args.model.path.lower() and (
            getattr(args, "max_new_tokens", 100) > 128
        ):
            layer_types = model.model.config.layer_types
            full_attn_layer_ids = [
                i for i, t in enumerate(layer_types) if t != "sliding_attention"
            ]
            n_layers = len(full_attn_layer_ids)
            n_heads = model.model.config.num_attention_heads
            model.model.config.num_hidden_layers = n_layers  # Patch the config to reflect the actual number of layers with attention
        else:
            n_layers = model.model.config.num_hidden_layers
            n_heads = model.model.config.num_attention_heads

        save_eval = getattr(args, "save_eval", False)
        alpha = 0.2

        run_all_versions = getattr(args, "run_all_versions", True)
        run_rauq_grid = getattr(args, "run_rauq_grid", True)
        if run_all_versions:
            rauq_config = {
                "aggregation": ["median", "max", "mean"],
                "token_aggregation": ["meanlog", "sumlog", "mean", "median"],
                "use_entropy": [True, False],
                "alpha": np.arange(0, 1.01, 0.1),
                "all_layers": [True, False],
                "head": ["max", "mean", "top5"],
            }
        else:
            rauq_config = {
                "aggregation": ["max", "mean"],
                "token_aggregation": ["meanlog"],
                "use_entropy": [True, False],
                "alpha": np.arange(0.1, 0.91, 0.1),
                "all_layers": [True, False],
                "head": ["max"],
            }

        if run_rauq_grid:
            # Get the keys and values
            keys = rauq_config.keys()
            values = rauq_config.values()

            # Iterate over all combinations
            for combination in itertools.product(*values):
                # Create a dictionary for this combination
                config = dict(zip(keys, combination))
                print(config)  # or do whatever you need with this config
                estimators += [
                    RAUQ(
                        n_layers=n_layers,
                        n_heads=n_heads,
                        aggregation=config["aggregation"],
                        token_aggregation=config["token_aggregation"],
                        alpha=config["alpha"],
                        use_entropy=config["use_entropy"],
                        all_layers=config["all_layers"],
                        head=config["head"],
                        print_alpha=True,
                    )
                ]

        if getattr(args, "run_alpha_ablation", False):
            for alpha_i in np.arange(0, 1.01, 0.1):
                estimators += [
                    RAUQ(
                        n_layers=n_layers,
                        n_heads=n_heads,
                        aggregation="max",
                        token_aggregation="meanlog",
                        alpha=alpha_i,
                        print_alpha=True,
                    ),
                    RAUQ(
                        n_layers=n_layers,
                        n_heads=n_heads,
                        aggregation="max",
                        token_aggregation="meanlog",
                        alpha=alpha_i,
                        print_alpha=True,
                        use_entropy=True,
                    ),
                ]

        if getattr(args, "run_token_aggregation_ablation", False):
            for token_aggregation in [
                "meanlog",
                "sumlog",
                "mean",
                "median",
                "max",
                "min",
            ]:
                estimators += [
                    RAUQ(
                        n_layers=n_layers,
                        n_heads=n_heads,
                        aggregation="max",
                        token_aggregation=token_aggregation,
                        alpha=alpha,
                        name_suffix="token_aggregation_ablation",
                    ),
                    RAUQ(
                        n_layers=n_layers,
                        n_heads=n_heads,
                        aggregation="max",
                        token_aggregation=token_aggregation,
                        alpha=alpha,
                        use_entropy=True,
                        name_suffix="token_aggregation_ablation",
                    ),
                ]

        if getattr(args, "run_layer_aggregation_ablation", False):
            for aggregation in ["median", "max", "mean", "meanmax", "medianmax"]:
                estimators += [
                    RAUQ(
                        n_layers=n_layers,
                        n_heads=n_heads,
                        aggregation=aggregation,
                        token_aggregation="meanlog",
                        alpha=alpha,
                        name_suffix="layer_aggregation_ablation",
                    ),
                    RAUQ(
                        n_layers=n_layers,
                        n_heads=n_heads,
                        aggregation=aggregation,
                        token_aggregation="meanlog",
                        alpha=alpha,
                        use_entropy=True,
                        name_suffix="layer_aggregation_ablation",
                    ),
                ]

        if getattr(args, "run_head_ablation", False):
            for head in ["max", "mean", "top5"]:
                estimators += [
                    RAUQ(
                        n_layers=n_layers,
                        n_heads=n_heads,
                        aggregation="max",
                        token_aggregation="meanlog",
                        alpha=alpha,
                        head=head,
                        name_suffix="head_ablation",
                    ),
                    RAUQ(
                        n_layers=n_layers,
                        n_heads=n_heads,
                        aggregation="max",
                        token_aggregation="meanlog",
                        alpha=alpha,
                        head=head,
                        use_entropy=True,
                        name_suffix="head_ablation",
                    ),
                ]

        if getattr(args, "run_layer_ablation", run_all_versions):
            for layer in list(range(n_layers)):
                estimators += [
                    RAUQ(
                        n_layers=n_layers,
                        n_heads=n_heads,
                        aggregation="max",
                        token_aggregation="meanlog",
                        alpha=alpha,
                        layers=[layer],
                        print_layer=True,
                    ),
                ]
            middle_layers = [
                layer for layer in [14, 15, 16, 17, 18] if layer < n_layers
            ]
            if middle_layers:
                estimators += [
                    RAUQ(
                        n_layers=n_layers,
                        n_heads=n_heads,
                        aggregation="max",
                        token_aggregation="meanlog",
                        alpha=alpha,
                        layers=middle_layers,
                        print_layer=True,
                    ),
                ]

        if getattr(args, "run_single_head", False):
            estimators += [
                RAUQ(
                    n_layers=n_layers,
                    n_heads=n_heads,
                    aggregation="max",
                    token_aggregation="meanlog",
                    alpha=alpha,
                    single_head=True,
                ),
            ]

        if getattr(args, "run_formula_ablation", False) or getattr(
            args, "run_formula_ablations", False
        ):
            for ablation in [
                "simple_rec",
                "no_rec",
                "no_attn",
                "multiply",
                "multiply_uq",
                "sum_uq",
            ]:
                estimators += [
                    RAUQ(
                        n_layers=n_layers,
                        n_heads=n_heads,
                        aggregation="max",
                        token_aggregation="meanlog",
                        alpha=alpha,
                        ablation=ablation,
                    ),
                    RAUQ(
                        n_layers=n_layers,
                        n_heads=n_heads,
                        aggregation="max",
                        token_aggregation="meanlog",
                        alpha=alpha,
                        ablation=ablation,
                        use_entropy=True,
                    ),
                ]

        # Delta-based ablations: sliding window average of attentions to t-1
        if getattr(args, "run_delta_ablations", False) or getattr(
            args, "run_delta_ablation", False
        ):
            for ablation_type in ["delta_sliding", "delta_ratio", "delta_rolling"]:
                for alpha_val in np.arange(0.1, 0.91, 0.1):
                    estimators += [
                        RAUQ(
                            n_layers=n_layers,
                            n_heads=n_heads,
                            aggregation="max",
                            token_aggregation="meanlog",
                            alpha=alpha_val,
                            ablation=ablation_type,
                            print_alpha=True,
                        ),
                    ]

        if save_eval:
            estimators += [
                RAUQ(
                    n_layers=n_layers,
                    n_heads=n_heads,
                    aggregation="max",
                    token_aggregation="meanlog",
                    alpha=alpha,
                    parameters_path=parameters_path,
                    save_eval=True,
                ),
            ]

    if args.use_ens_ue:
        if not (model.model_type == "Seq2SeqLM"):
            raise NotImplementedError(
                "Only Encoder-Decoder models can be ensembled at this time"
            )

        token_measures = all_token_estimators()
        if args.model.ensembling_mode == "pe":
            sequence_measures = all_pe_estimators()
        elif args.model.ensembling_mode == "ep":
            sequence_measures = all_ep_estimators()
        else:
            raise ValueError(
                f'Ensemble type should be one of: "pe", "ep", but is {args.ens_type} instead'
            )
        estimators += token_measures + sequence_measures

    if args.use_tok_ue:
        estimators += [
            MaximumTokenProbability(),
            TokenEntropy(),
            PointwiseMutualInformation(),
            ConditionalPointwiseMutualInformation(),
            SemanticEntropyToken(model.model_path, args.cache_path),
        ]

    if getattr(args, "use_claim_ue", False):
        estimators += [
            MaximumClaimProbability(),
            PerplexityClaim(),
            MaxTokenEntropyClaim(),
            PointwiseMutualInformationClaim(),
            PTrueClaim(),
            ClaimConditionedProbabilityClaim(nli_context="no_context"),
            ClaimConditionedProbabilityClaim(nli_context="fact_pref"),
        ]

    additional_estimators = getattr(args, "additional_estimators", {})
    additional_estimators_kwargs = getattr(args, "additional_estimators_kwargs", {})

    for i, (module_name, estimator_classes) in enumerate(additional_estimators.items()):
        module = importlib.import_module(module_name)
        for j, estimator_class in enumerate(estimator_classes):
            try:
                estimator_kwargs = additional_estimators_kwargs[estimator_class]
            except KeyError:
                raise TypeError(f"Arguments for {estimator} were not passed")

            estimators.append(getattr(module, estimator_class)(**estimator_kwargs))

    return estimators


def get_generation_metrics(args):
    generation_metrics = getattr(args, "generation_metrics", None)
    ignore_regex = getattr(args, "source_ignore_regex", None)

    metric_name = getattr(args, "target_train_metric", "AlignScore")
    if metric_name == "AlignScoreMean":
        alignscorer = AlignScore(batch_size=4, return_mean=True)
    elif metric_name == "AlignScoreInv":
        alignscorer = AlignScore(
            batch_size=4,
            target_is_claims=False,
            source_ignore_regex=ignore_regex,
            source_as_target=True,
        )
    else:
        alignscorer = AlignScore(batch_size=4)

    if not generation_metrics:
        result = [
            RougeMetric("rougeL"),
            AccuracyMetric(
                target_ignore_regex=getattr(args, "target_ignore_regex", None),
                output_ignore_regex=getattr(args, "output_ignore_regex", None),
                normalize=getattr(args, "normalize", False),
            ),
            alignscorer,
        ]
        if args.task == "nmt":
            ignore_regex = getattr(args, "source_ignore_regex", None)
            result += [Comet(source_ignore_regex=ignore_regex)]
        if not getattr(args, "multiref", False):
            pass
        else:
            # Wrap each metric in AggregatedMetric
            result = [AggregatedMetric(base_metric=metric) for metric in result]
    else:
        result = []
        for metric in generation_metrics:
            metric_name = metric["name"]
            if getattr(args, "multiref", False) and metric_name == "BartScoreSeqMetric":
                raise ValueError("BartScoreSeqMetric does not support multiref")
            metric_class = globals()[metric_name]
            result.append(metric_class(*metric.get("args", [])))
    return result


def get_model_kwargs(args):
    model_kwargs = {}
    if getattr(args.model, "attn_implementation", None) is not None:
        model_kwargs["attn_implementation"] = args.model.attn_implementation
    if getattr(args.model, "use_cache", None) is not None:
        model_kwargs["use_cache"] = args.model.use_cache
    if getattr(args.model, "cache_implementation", None) is not None:
        model_kwargs["cache_implementation"] = args.model.cache_implementation
    if getattr(args.model, "dtype", None) is not None:
        if args.model.dtype == "bfloat16":
            model_kwargs["torch_dtype"] = torch.bfloat16

    return model_kwargs


def get_stat_calculator_names(config):
    model_type_raw = getattr(config.model, "type", "Whitebox")
    model_type = (
        "Blackbox"
        if model_type_raw == "Blackbox"
        else "VisualLM" if model_type_raw == "VisualLM" else "Whitebox"
    )
    language = getattr(config, "language", "en")
    output_attentions = getattr(config, "output_attentions", True) and (
        getattr(config.model, "type", "Whitebox") != "vLLMCausalLM"
    )
    output_hidden_states = (
        False if getattr(config.model, "type", "Whitebox") == "vLLMCausalLM" else True
    )
    hf_cache = getattr(config, "hf_cache", None)
    deberta_batch_size = getattr(config, "deberta_batch_size", 10)
    blackbox_supports_logprobs = model_type == "Blackbox" and getattr(
        config.model, "supports_logprobs", False
    )
    samples_n = getattr(config, "generation_params", {}).get("samples_n", 10)
    max_input_length = getattr(config, "max_input_length", 2048)
    all_stat_calculators = []
    stat_calculators = getattr(config, "stat_calculators", ["auto"])
    if "auto" in stat_calculators:
        all_stat_calculators += register_default_stat_calculators(
            model_type,
            language,
            hf_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            blackbox_supports_logprobs=blackbox_supports_logprobs,
            deberta_batch_size=deberta_batch_size,
            samples_n=samples_n,
            max_input_length=max_input_length,
        )

    has_train_stat_calculator = False
    for stat_calculator in stat_calculators:
        if stat_calculator == "auto":
            continue
        stats = list(stat_calculator.stats)
        if stat_calculator.name == "TrainingStatisticExtractionCalculator":
            has_train_stat_calculator = True
            if "train_attention_all" not in stats:
                stats.append("train_attention_all")
        sc = StatCalculatorContainer(
            name=stat_calculator.name,
            cfg=stat_calculator.cfg,
            stats=stats,
            dependencies=stat_calculator.dependencies,
            builder=stat_calculator.builder,
        )
        all_stat_calculators.append(sc)

    needs_train_stat_calculator = getattr(config, "run_single_head", False) or getattr(
        config, "run_supervised_baselines", False
    )
    if needs_train_stat_calculator and not has_train_stat_calculator:
        train_stat_cfg = {
            "dataset": config.dataset,
            "text_column": config.text_column,
            "label_column": getattr(config, "label_column", None),
            "prompt": getattr(config, "prompt", ""),
            "description": getattr(config, "description", ""),
            "few_shot_prompt": getattr(config, "few_shot_prompt", ""),
            "few_shot_split": getattr(config, "few_shot_split", "train"),
            "train_split": getattr(config, "train_split", "train"),
            "load_from_disk": getattr(config, "load_from_disk", False),
            "subsample_train_dataset": getattr(config, "subsample_train_dataset", -1),
            "batch_size": config.batch_size,
            "seed": config.seed,
            "size": 10000,
            "max_new_tokens": getattr(config, "max_new_tokens", 100),
            "background_train_dataset": getattr(
                config, "background_train_dataset", None
            ),
            "background_train_dataset_text_column": getattr(
                config, "background_train_dataset_text_column", None
            ),
            "background_train_dataset_label_column": getattr(
                config, "background_train_dataset_label_column", None
            ),
            "background_train_dataset_data_files": getattr(
                config, "background_train_dataset_data_files", None
            ),
            "background_load_from_disk": getattr(
                config, "background_load_from_disk", False
            ),
            "subsample_background_train_dataset": getattr(
                config, "subsample_background_train_dataset", -1
            ),
            "output_attentions": output_attentions,
            "output_hidden_states": output_hidden_states,
            "return_embeddings": getattr(config, "run_supervised_baselines", False),
            "return_token_embeddings": getattr(
                config, "run_supervised_baselines", False
            ),
            "return_lookback_ratios": getattr(
                config, "run_supervised_baselines", False
            ),
            "return_attention_features": getattr(
                config, "run_supervised_baselines", False
            ),
            "target_metric": None,
            "is_ood_exps": getattr(config, "is_ood_exps", False),
        }
        k_ds = 1
        while getattr(config, f"train_dataset_{k_ds}", False):
            for key in [
                "train_dataset",
                "train_text_column",
                "train_label_column",
                "train_prompt",
                "train_description",
                "train_few_shot_prompt",
                "train_n_shot",
                "few_shot_split",
                "train_split",
                "max_new_tokens",
            ]:
                attr = f"{key}_{k_ds}"
                if hasattr(config, attr):
                    train_stat_cfg[attr] = getattr(config, attr)
            k_ds += 1

        train_stats = ["train_attention_all", "train_greedy_log_likelihoods"]
        if getattr(config, "run_supervised_baselines", False):
            train_stats = [
                "train_embeddings",
                "train_token_embeddings",
                "background_train_token_embeddings",
                "train_greedy_log_likelihoods",
                "train_lookback_ratios",
                "train_attention_features",
                "train_metrics",
            ]

        all_stat_calculators.append(
            StatCalculatorContainer(
                name="TrainingStatisticExtractionCalculator",
                cfg=OmegaConf.create(train_stat_cfg),
                stats=train_stats,
                dependencies=[],
                builder="default_TrainingStatisticExtractionCalculator",
            )
        )

    return all_stat_calculators


if __name__ == "__main__":
    main()
