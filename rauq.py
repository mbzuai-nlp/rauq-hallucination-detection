import os
import numpy as np

from typing import Dict

from lm_polygraph.estimators.estimator import Estimator


class RAUQ(Estimator):
    def __init__(
        self,
        alpha: float = 0.2,
        n_layers: int = 32,
        n_heads: int = 32,
        all_layers: bool = False,
        use_entropy: bool = False,
        layers=None,
        print_layer=False,
        aggregation: str = "mean",
        token_aggregation: str = "meanmin",
        head: str = "max",
        ablation: str = None,
        save_eval: bool = False,
        parameters_path: str = "",
        print_alpha: bool = False,
        single_head: bool = False,
        name_suffix: str = None,
    ):
        deps = ["attention_all", "greedy_log_likelihoods"]
        if single_head:
            deps += ["train_attention_all", "train_greedy_log_likelihoods"]
        if use_entropy:
            super().__init__(deps + ["entropy"], "sequence")
        else:
            super().__init__(deps, "sequence")

        self.print_layer = print_layer
        self.alpha = alpha
        self.token_aggregation = token_aggregation
        self.aggregation = aggregation
        self.use_entropy = use_entropy
        self.single_head = single_head
        self.name_suffix = name_suffix
        self.selected_head = None
        if self.single_head:
            self.is_fitted = False

        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head = head

        self.all_layers = all_layers
        self.ablation = ablation

        if layers is None:
            self.layers = (
                list(range(n_layers))
                if self.all_layers
                else list(range(n_layers // 3, int(np.ceil(n_layers / 3 * 2) + 1)))
            )
        else:
            self.layers = layers
        self.print_alpha = print_alpha

        self.save_eval = save_eval
        self.eval_index = 0
        if len(parameters_path):
            self_name = self.__str__().replace(" ", "_")
            self.full_path = f"{parameters_path}/{self_name}"
            os.makedirs(self.full_path, exist_ok=True)

    def __str__(self):
        method_desc = ""
        method_desc += f" {self.aggregation}_{self.token_aggregation}_{self.head}"
        if self.use_entropy:
            method_desc += f" Entropy"
        if self.ablation is not None:
            method_desc += f" {self.ablation}"
        if self.all_layers:
            method_desc += f" all_layers"
        if self.print_alpha:
            method_desc += f" {self.alpha:.2f}"
        if self.print_layer:
            method_desc += " layers=" + ",".join(str(layer) for layer in self.layers)
        if self.single_head:
            method_desc += " single_head"
        if self.name_suffix is not None:
            method_desc += f" {self.name_suffix}"
        return f"RAUQ{method_desc}"

    def _prev_token_attentions(self, attention_all):
        attentions = []
        for attention_weight in attention_all:
            # Reshape attention weights to separate layers and heads
            reshaped_weights = attention_weight.reshape(
                self.n_layers,
                self.n_heads,
                attention_weight.shape[-2],
                attention_weight.shape[-1],
            )
            # Extract attention weights for previous token with offset -1
            attenion_prev_token = np.diagonal(
                reshaped_weights, offset=-1, axis1=2, axis2=3
            )
            attentions.append(attenion_prev_token)
        return attentions

    def attention_selection(self, attentions, j, layer, head):
        if self.head == "max":
            # attn = attentions[j-1, layer, head]
            attn = attentions[layer, head, j - 1]
        elif self.head == "mean":
            # attn = attentions[j-1, layer].mean(-1)
            attn = attentions[layer, :, j - 1].mean(-1)
        elif self.head == "top5":
            # attn = attentions[j-1, layer, head].mean(-1)
            attn = attentions[layer, head, j - 1].mean(-1)
        else:
            raise NotImplementedError
        return attn

    def tokens_aggregation(
        self, conf_scores, attentions, log_probabilities, layer, head
    ):
        if self.ablation == "multiply_uq":
            return np.mean(log_probabilities) * np.mean(
                np.log(attentions[layer, head, :])
            )
        elif self.ablation == "sum_uq":
            return -(
                np.mean(log_probabilities) + np.mean(np.log(attentions[layer, head, :]))
            )

        if self.token_aggregation == "meanmin":
            uq = 1 - (np.mean(conf_scores) + np.min(conf_scores)) / 2
        elif self.token_aggregation == "mean":
            uq = 1 - np.mean(conf_scores)
        elif self.token_aggregation == "min":
            uq = 1 - np.min(conf_scores)
        elif self.token_aggregation == "max":
            uq = 1 - np.max(conf_scores)
        elif self.token_aggregation == "median":
            uq = 1 - np.median(conf_scores)
        elif self.token_aggregation == "meanlog":
            uq = 1 - np.log(conf_scores).mean()
        elif self.token_aggregation == "sumlog":
            uq = 1 - np.log(conf_scores).sum()
        else:
            raise NotImplementedError
        return uq

    def layers_aggregation(self, uq_scores_layers):
        if self.aggregation == "mean":
            uq = np.mean(uq_scores_layers)
        elif self.aggregation == "median":
            uq = np.median(uq_scores_layers)
        elif self.aggregation == "max":
            uq = np.max(uq_scores_layers)
        elif self.aggregation == "meanmax":
            uq = (np.mean(uq_scores_layers) + np.max(uq_scores_layers)) / 2
        elif self.aggregation == "medianmax":
            uq = (np.median(uq_scores_layers) + np.max(uq_scores_layers)) / 2
        else:
            raise NotImplementedError
        return uq

    def ablation_formula(self, p_j, p_jm1, logprob_jm1, attn, attn_t1_history=None):
        if self.ablation == "simple_rec":
            score = self.alpha * p_j + (1 - self.alpha) * attn * np.exp(logprob_jm1)
        elif self.ablation == "no_rec":
            score = self.alpha * p_j + (1 - self.alpha) * attn
        elif self.ablation == "no_attn":
            score = self.alpha * p_j + (1 - self.alpha) * p_jm1
        elif self.ablation == "multiply":
            score = p_j * attn
        elif self.ablation == "delta_sliding":
            # Delta-based: attention minus sliding average of all previous attentions
            sliding_avg = np.mean(attn_t1_history) if len(attn_t1_history) > 0 else attn
            delta_attn = attn - sliding_avg
            score = self.alpha * p_j + (1 - self.alpha) * delta_attn * p_jm1
        elif self.ablation == "delta_ratio":
            # Ratio-based: attention divided by sliding average
            sliding_avg = np.mean(attn_t1_history) if len(attn_t1_history) > 0 else attn
            delta_attn = attn / (sliding_avg + 1e-10)
            score = self.alpha * p_j + (1 - self.alpha) * delta_attn * p_jm1
        elif self.ablation == "delta_rolling":
            # Rolling window delta (last K tokens only)
            window_size = 5
            window = (
                attn_t1_history[-window_size:]
                if len(attn_t1_history) >= window_size
                else attn_t1_history
            )
            rolling_avg = np.mean(window) if len(window) > 0 else attn
            delta_attn = attn - rolling_avg
            score = self.alpha * p_j + (1 - self.alpha) * delta_attn * p_jm1
        else:
            score = 0  # aggregation of sequence-level scores
        return score

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:

        # attention_features_values = stats[f"attention_features_values"]
        # attention_features_values = [np.array(item) for sublist in attention_features_values for item in sublist]
        attentions = self._prev_token_attentions(stats["attention_all"])

        if self.single_head and self.selected_head is None:
            train_attentions = self._prev_token_attentions(stats["train_attention_all"])
            train_attentions = [attn for attn in train_attentions if attn.shape[-1] > 0]
            if len(train_attentions):
                self.selected_head = (
                    np.concatenate(train_attentions, axis=-1).mean(-1).argmax(-1)
                )
            else:
                self.selected_head = np.zeros(self.n_layers, dtype=int)
            self.is_fitted = True

        greedy_log_likelihoods = stats["greedy_log_likelihoods"]
        if self.use_entropy:
            entropy = stats["entropy"]
            vocab_size = len(stats["greedy_log_probs"][0][0])
            max_entropy = np.log(vocab_size)

        k = 0
        uq_scores = []
        heads = []
        for idx in range(len(greedy_log_likelihoods)):

            # attentions = np.array([attention_features_values[ind][0] for ind in range(k, k+len(greedy_log_likelihoods[idx])-1)]) # zero means use of the attention only on previous token
            # attentions = attentions.reshape(-1, self.n_layers, self.n_heads)
            if self.use_entropy:
                log_probabilities = np.log(max_entropy - np.array(entropy[idx]) + 1e-10)
            else:
                log_probabilities = greedy_log_likelihoods[idx]

            uq_scores_layers = []
            heads_layers = []
            for layer in self.layers:
                # if self.use_entropy:
                #     p_i = [log_probabilities[0]]
                # else:
                p_i = [np.exp(log_probabilities[0])]
                # head = attentions.mean(0)[layer].argmax() # select the most attentive head
                head = attentions[idx][layer].mean(-1).argmax()
                if self.head == "top5":
                    # head = attentions.mean(0)[layer].argsort()[-5:]
                    head = attentions[idx][layer].mean(-1).argsort()[-5:]
                if self.single_head:
                    head = self.selected_head[layer]
                heads_layers.append(head)
                attn_t1_history = (
                    []
                )  # Track attention history for delta-based ablations
                for j in range(1, len(log_probabilities)):
                    # if self.use_entropy:
                    #     p_j = log_probabilities[j]
                    # else:
                    p_j = np.exp(log_probabilities[j])

                    p_jm1 = p_i[-1]

                    attn = self.attention_selection(attentions[idx], j, layer, head)

                    # For delta-based ablations, track attention history
                    if self.ablation in [
                        "delta_sliding",
                        "delta_ratio",
                        "delta_rolling",
                    ]:
                        attn_t1_history.append(attn)
                        conf = self.ablation_formula(
                            p_j, p_jm1, log_probabilities[j - 1], attn, attn_t1_history
                        )
                    elif self.ablation is None:
                        conf = self.alpha * p_j + (1 - self.alpha) * attn * p_jm1
                    else:
                        # if self.use_entropy:
                        #     conf = self.ablation_formula(p_j, p_jm1, np.log(log_probabilities[j-1]), attn)
                        # else:
                        conf = self.ablation_formula(
                            p_j, p_jm1, log_probabilities[j - 1], attn
                        )
                    p_i.append(conf)

                uq = self.tokens_aggregation(
                    p_i, attentions[idx], log_probabilities, layer, head
                )
                uq_scores_layers.append(uq)
            heads.append(heads_layers)
            uq_scores.append(self.layers_aggregation(uq_scores_layers))
            k += len(log_probabilities) - 1

            if self.save_eval:
                np.save(
                    f"{self.full_path}/attentions_{self.eval_index}.npy",
                    np.array(attentions),
                )
                np.save(
                    f"{self.full_path}/log_probs_{self.eval_index}.npy",
                    np.array(greedy_log_likelihoods),
                )
                np.save(
                    f"{self.full_path}/heads_{self.eval_index}.npy", np.array(heads)
                )
                self.eval_index += 1

        return np.array(uq_scores)
