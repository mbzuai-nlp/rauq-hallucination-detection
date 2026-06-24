import os
import numpy as np
import torch
import itertools
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

from typing import Dict

from lm_polygraph.estimators.estimator import Estimator

import pickle
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge
from lm_polygraph.generation_metrics.alignscore import AlignScore
from lm_polygraph.generation_metrics.aggregated_metric import AggregatedMetric
from lm_polygraph.ue_metrics import PredictionRejectionArea

from datasets import load_dataset
from collections import defaultdict
import pickle
import random
from transformers import AutoConfig, AutoTokenizer
import numpy as np
import math
from tqdm import tqdm


class LLMCheckAttention(Estimator):
    def __init__(
        self,
        layer=16,
        aggregation="sum",
        fix=False,
        gen_only=False,
        one_head=False,
        save_eval: bool = False,
        save_eval_full: bool = False,
        parameters_path: str = "",
    ):
        super().__init__(["forwardpass_attention_weights"], "sequence")
        self.layer = layer
        self.aggregation = aggregation
        self.one_head = one_head
        self.fix = fix
        self.gen_only = gen_only
        self.save_eval = save_eval
        self.save_eval_full = save_eval_full
        self.eval_index = 0
        if len(parameters_path):
            self_name = self.__str__().replace(" ", "_")
            self.full_path = f"{parameters_path}/{self_name}"
            os.makedirs(self.full_path, exist_ok=True)

    def __str__(self):
        one_head = "_one_head" if self.one_head else ""
        if self.fix:
            return (
                f"LLMCheckAttentionFIX Layer {self.layer}, {self.aggregation}{one_head}"
            )
        elif self.gen_only:
            return (
                f"LLMCheckAttentionGEN Layer {self.layer}, {self.aggregation}{one_head}"
            )
        return f"LLMCheckAttention Layer {self.layer}, {self.aggregation}{one_head}"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        forwardpass_attention_weights = stats["forwardpass_attention_weights"]
        greedy_tokens = stats["greedy_tokens"]

        ue = []
        for k, attention_weight in enumerate(forwardpass_attention_weights):
            ue_i = 0
            if self.one_head:
                ue_i_array = []
                vals_i_array = []
                for attn in attention_weight[self.layer]:
                    if self.fix:
                        attn = attn[:-1, :-1]  # USE ONLY GENERATED TOKENS
                    elif self.gen_only:
                        attn = attn[
                            -len(greedy_tokens[k]) : -1, -len(greedy_tokens[k]) : -1
                        ]  # USE ONLY GENERATED TOKENS
                    if self.aggregation == "sum":
                        ue_i_array.append(np.sum(np.log(np.diag(attn))))
                    elif self.aggregation == "mean":
                        ue_i_array.append(np.mean(np.log(np.diag(attn))))

                    vals_i_array.append(np.mean(np.diag(attn)))

                max_ind = np.argmax(vals_i_array)
                ue_i = ue_i_array[max_ind]

            else:
                for attn in attention_weight[self.layer]:
                    if self.fix:
                        attn = attn[:-1, :-1]  # USE ONLY GENERATED TOKENS
                    elif self.gen_only:
                        attn = attn[
                            -len(greedy_tokens[k]) : -1, -len(greedy_tokens[k]) : -1
                        ]  # USE ONLY GENERATED TOKENS

                    if self.aggregation == "sum":
                        ue_i += np.sum(np.log(np.diag(attn)))
                    elif self.aggregation == "mean":
                        ue_i += np.mean(np.log(np.diag(attn)))

                ue_i /= len(attention_weight[self.layer])
            ue.append(ue_i)

        if self.save_eval:
            first_attention = []
            for attention_weight, tokens in zip(
                forwardpass_attention_weights, greedy_tokens
            ):
                attention_weight = np.asarray(attention_weight)
                first_attention.append(
                    attention_weight[:, :, -len(tokens) - 1, -len(tokens) - 1]
                )
            np.save(
                f"{self.full_path}/first_attention_{self.eval_index}.npy",
                np.array(first_attention),
            )
            if self.save_eval_full:
                np.save(
                    f"{self.full_path}/forwardpass_attention_weights_{self.eval_index}.npy",
                    np.array(forwardpass_attention_weights),
                )
            self.eval_index += 1

        return -np.array(ue)
