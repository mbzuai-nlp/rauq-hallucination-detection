import codecs

from lm_polygraph.stat_calculators.statistic_extraction import (
    TrainingStatisticExtractionCalculator,
)

from utils.dataset import Dataset


def _decode_config_text(value):
    return codecs.decode(value, "unicode_escape") if isinstance(value, str) else value


def _seed(config):
    return config.seed if isinstance(config.seed, int) else list(config.seed)[0]


def _normalize_targets(dataset):
    if len(dataset.y) and isinstance(dataset.y[0], list):
        dataset.y = [y[0] for y in dataset.y]


def _load_single_train_dataset(config):
    return Dataset.load(
        config.dataset,
        config.text_column,
        getattr(config, "label_column", None),
        batch_size=config.batch_size,
        prompt=getattr(config, "prompt", ""),
        description=getattr(config, "description", ""),
        few_shot_prompt=getattr(config, "few_shot_prompt", ""),
        n_shot=getattr(config, "n_shot", 5),
        few_shot_split=getattr(config, "few_shot_split", "train"),
        split=config.train_split,
        size=getattr(config, "size", 10000),
        load_from_disk=config.load_from_disk,
        max_new_tokens=getattr(config, "max_new_tokens", 100),
    )


def _load_ood_train_dataset(config):
    train_dataset = None
    k_ds = 1
    while getattr(config, f"train_dataset_{k_ds}", False):
        train_dataset_k = Dataset.load(
            getattr(config, f"train_dataset_{k_ds}"),
            getattr(config, f"train_text_column_{k_ds}"),
            getattr(config, f"train_label_column_{k_ds}"),
            batch_size=config.batch_size,
            prompt=_decode_config_text(getattr(config, f"train_prompt_{k_ds}", "")),
            description=_decode_config_text(
                getattr(config, f"train_description_{k_ds}", "")
            ),
            few_shot_prompt=_decode_config_text(
                getattr(config, f"train_few_shot_prompt_{k_ds}", "")
            ),
            mmlu_max_subject_size=getattr(config, "mmlu_max_subject_size", 100),
            n_shot=getattr(config, f"train_n_shot_{k_ds}", 5),
            few_shot_split=getattr(config, f"few_shot_split_{k_ds}", "train"),
            split=getattr(config, f"train_split_{k_ds}", "train"),
            max_new_tokens=getattr(config, f"max_new_tokens_{k_ds}", 100),
            size=getattr(config, "size", 10000),
            load_from_disk=config.load_from_disk,
        )
        if config.subsample_train_dataset != -1:
            train_dataset_k.subsample(
                config.subsample_train_dataset, seed=_seed(config)
            )
        _normalize_targets(train_dataset_k)
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


def _load_background_train_dataset(config):
    if not getattr(config, "background_train_dataset", None):
        return None
    dataset = Dataset.load(
        config.background_train_dataset,
        config.background_train_dataset_text_column,
        config.background_train_dataset_label_column,
        batch_size=config.batch_size,
        data_files=getattr(config, "background_train_dataset_data_files", None),
        split="train",
        size=100000,
        load_from_disk=getattr(config, "background_load_from_disk", False),
        max_new_tokens=getattr(config, "max_new_tokens", 100),
    )
    if config.subsample_background_train_dataset != -1:
        dataset.subsample(config.subsample_background_train_dataset, seed=_seed(config))
    return dataset


def load_stat_calculator(config, builder):
    if getattr(config, "is_ood_exps", False) and getattr(
        config, "train_dataset_1", False
    ):
        train_dataset = _load_ood_train_dataset(config)
    else:
        train_dataset = _load_single_train_dataset(config)
        if config.subsample_train_dataset != -1:
            train_dataset.subsample(config.subsample_train_dataset, seed=_seed(config))
        _normalize_targets(train_dataset)

    target_metric = getattr(config, "target_metric", None)
    if target_metric is None and hasattr(builder, "generation_metrics"):
        target_metric = builder.generation_metrics[-1]

    return TrainingStatisticExtractionCalculator(
        train_dataset=train_dataset,
        background_train_dataset=_load_background_train_dataset(config),
        output_attentions=config.output_attentions,
        output_hidden_states=config.output_hidden_states,
        return_embeddings=getattr(config, "return_embeddings", False),
        return_token_embeddings=getattr(config, "return_token_embeddings", False),
        return_lookback_ratios=getattr(config, "return_lookback_ratios", False),
        return_attention_features=getattr(config, "return_attention_features", False),
        target_metric=target_metric,
    )
