import numpy as np
from lm_polygraph.estimators.estimator import Estimator
from typing import Dict
from captum.attr import (
    LayerIntegratedGradients,
    LLMGradientAttribution,
    TextTokenInput,
)
import torch


class IntegratedGradients(Estimator):
    def __init__(self, model, model_name, positive=False, reccurent=False):
        super().__init__(["greedy_tokens"], "sequence")
        if model_name in ["facebook/opt-350m"]:
            self.lig = LayerIntegratedGradients(
                model.model, model.model.base_model.decoder.embed_tokens
            )
        elif "llama" in model_name.lower():
            self.lig = LayerIntegratedGradients(
                model.model, model.model.model.embed_tokens
            )
        else:
            self.lig = LayerIntegratedGradients(
                model.model, model.model.base_model.word_embeddings
            )

        self.llm_attr = LLMGradientAttribution(self.lig, model.tokenizer)
        self.positive = positive
        self.reccurent = reccurent

    def __str__(self):
        pos = "positive" if self.positive else ""
        rec = ", reccurent" if self.reccurent else ""
        return f"IntegratedGradients {pos}{rec}"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        tokenizer = stats["tokenizer"]
        input_tokens = torch.tensor(stats["input_tokens"][0])
        greedy_tokens = torch.tensor(stats["greedy_tokens"][0])

        possible_len = 10
        if len(input_tokens) > possible_len:
            input_tokens = input_tokens[len(input_tokens) - possible_len :]

        int_grads_tensor = torch.zeros(len(greedy_tokens), len(greedy_tokens))

        for i in range(1, len(greedy_tokens)):
            eval_prompt_ids = torch.cat(
                (
                    input_tokens,
                    greedy_tokens[:i],
                ),
            )
            eval_prompt = tokenizer.decode(eval_prompt_ids, skip_special_tokens=True)

            target = greedy_tokens[i : i + 1]
            token_attr = self.get_attribution(eval_prompt, target, tokenizer)
            cut_off = token_attr[0].shape[0]
            int_grads_tensor[i, :i] = token_attr[0][cut_off - i :]

        greedy_log_likelihoods = stats["greedy_log_likelihoods"]
        int_grads_tensor = int_grads_tensor[None, :]
        int_grads_tensor = int_grads_tensor.cpu().numpy()
        if self.positive:
            int_grads_tensor[int_grads_tensor < 0] = 0

        focus_ue = []
        lig_features = []
        for int_grad, greedy_log_likelihood in zip(
            int_grads_tensor, greedy_log_likelihoods
        ):
            lig_features.append([])
            weight = int_grad / (np.sum(int_grad, axis=1, keepdims=True) + 1e-6)
            if self.reccurent:
                token_focus = []
                for i, token_weights in enumerate(weight):
                    ue = greedy_log_likelihood[i]
                    if len(token_focus):
                        ue += (
                            np.array(token_focus) * token_weights[: len(token_focus)]
                        ).sum()
                    token_focus.append(ue)
                focus_ue.append(-np.mean(token_focus))
            else:
                token_focus = (np.array(greedy_log_likelihood)[None, :] * weight).sum(0)
                focus_ue.append(-(token_focus + np.array(greedy_log_likelihood)).mean())

            for i, token_weights in enumerate(weight):
                if i:
                    lig_features[-1].append(token_weights[i - 1])
        stats["lig_features"] = lig_features

        return np.array(focus_ue)

    def get_attribution(self, eval_prompt, target, tokenizer):
        try:
            inp = TextTokenInput(
                eval_prompt,
                tokenizer,
                # skip_tokens=[1],  # skip the special token for the start of the text <s>
            )
            attr_res = self.llm_attr.attribute(inp, target=target)
            token_attr = attr_res.token_attr.cpu()
        except:
            print(eval_prompt, target)

        return token_attr
