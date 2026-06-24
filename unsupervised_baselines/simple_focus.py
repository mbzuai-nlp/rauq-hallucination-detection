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


class SimpleFocus(Estimator):
    def __init__(self, reccurent=False, only_prev=False, one_head=False, layer=16):
        super().__init__(["attention_all", "greedy_log_likelihoods"], "sequence")
        self.reccurent = reccurent
        self.only_prev = only_prev
        self.one_head = one_head
        self.layer = layer

    def __str__(self):
        rec = "reccurent" if self.reccurent else ""
        only_prev = " only_prev" if self.only_prev else ""
        one_head = f" one_head_{self.layer}" if self.one_head else ""
        return f"SimpleFocus {rec}{only_prev}{one_head}"

    def entropy(self, p):
        p_torch = torch.tensor(p)
        return torch.sum(
            -torch.where(p_torch > 0, p_torch * p_torch.log2(), p_torch.new([0.0])),
            dim=-1,
        ).numpy()

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        # take the embeddings
        attention_weights = []
        for weights in stats["attention_all"]:
            if weights.ndim == 3:
                # (n, seq_len, seq_len) -> (seq_len, seq_len)
                attention_weights.append(np.max(weights, axis=0))
            elif weights.ndim == 2:
                # (seq_len, seq_len)
                attention_weights.append(weights)
            elif weights.ndim == 4:
                # (num_layers, num_heads, seq_len, seq_len)
                attention_weights.append(np.max(weights, axis=(0, 1)))
            else:
                print(f"Unexpected attention weights shape: {weights.shape}")
                attention_weights.append(np.array([]))
        greedy_log_likelihoods = stats["greedy_log_likelihoods"]

        if self.one_head:
            forwardpass_attention_weights = stats["forwardpass_attention_weights"]
            greedy_tokens = stats["greedy_tokens"]

        focus_ue = []
        for k, (attention_weight, greedy_log_likelihood) in enumerate(
            zip(attention_weights, greedy_log_likelihoods)
        ):

            if self.one_head:
                attention_weight_l = forwardpass_attention_weights[k][
                    :, :, -len(greedy_tokens[k]) : -1, -len(greedy_tokens[k]) : -1
                ][
                    self.layer
                ]  # select attention weights on gen. tokens
                head = attention_weight_l.mean((1, 2)).argmax()  # select head
                attention_weight_l = attention_weight_l[head]
                attention_weight_l = np.pad(attention_weight_l, (1, 1))[
                    :-1, 1:
                ]  # pad first row and last column with zeros
                weight = attention_weight_l / (
                    np.sum(attention_weight_l, axis=1, keepdims=True) + 1e-6
                )
            else:
                weight = attention_weight / (
                    np.sum(attention_weight, axis=1, keepdims=True) + 1e-6
                )
            if self.reccurent:
                token_focus = []
                for i, token_weights in enumerate(weight):
                    ue = greedy_log_likelihood[i]
                    if len(token_focus):
                        if self.only_prev:
                            coef = token_weights[: len(token_focus)][-1]
                            ue = (
                                greedy_log_likelihood[i] * (1 - coef)
                                + np.array(token_focus)[-1] * coef
                            )
                        else:
                            ue += (
                                np.array(token_focus)
                                * token_weights[: len(token_focus)]
                            ).sum()
                    token_focus.append(ue)
                focus_ue.append(-np.mean(token_focus))
            else:
                token_focus = (np.array(greedy_log_likelihood)[None, :] * weight).sum(0)
                focus_ue.append(-(token_focus + np.array(greedy_log_likelihood)).mean())

        return np.array(focus_ue)
