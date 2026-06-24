"""
Patch to apply extract_final_answer after tokenizer decode in lm_polygraph.

This patch monkey-patches the tokenizer's decode method and stat calculators
to apply extract_final_answer to the result after each decode call.
"""

import logging
import sys
from functools import wraps

log = logging.getLogger("lm_polygraph.tokenizer_patch")


def patch_tokenizer_decode(extract_final_answer_func):
    """
    Patch tokenizer.decode to apply extract_final_answer after decoding.

    Args:
        extract_final_answer_func: A function that takes a string and returns
                                   the extracted final answer.
    """
    from lm_polygraph.utils.model import WhiteboxModel

    # Store the original decode method
    if not hasattr(WhiteboxModel, "_original_decode"):
        WhiteboxModel._original_decode = WhiteboxModel.tokenizer.decode

    def patched_decode(token_ids, *args, **kwargs):
        result = WhiteboxModel._original_decode(token_ids, *args, **kwargs)
        return extract_final_answer_func(result)

    # Monkey-patch the tokenizer's decode method
    WhiteboxModel.tokenizer.decode = patched_decode


def patch_stat_calculators(extract_final_answer_func):
    """
    Patch GreedyProbs and Sample stat calculators to post-process their text outputs.

    Args:
        extract_final_answer_func: A function that takes a string and returns
                                   the extracted final answer.
    """
    try:
        from lm_polygraph.stat_calculators import greedy_probs

        if hasattr(greedy_probs, "GreedyProbs"):
            original_calculate = greedy_probs.GreedyProbs.calculate

            @wraps(original_calculate)
            def patched_calculate(self, texts, model, **kwargs):
                result = original_calculate(self, texts, model, **kwargs)
                if "greedy_texts" in result:
                    result["greedy_texts"] = [
                        extract_final_answer_func(text)
                        for text in result["greedy_texts"]
                    ]
                return result

            greedy_probs.GreedyProbs.calculate = patched_calculate
            log.info("Patched GreedyProbs.calculate")
    except Exception as e:
        log.warning(f"Failed to patch GreedyProbs: {e}")

    try:
        from lm_polygraph.stat_calculators import sample

        if hasattr(sample, "Sample"):
            original_calculate = sample.Sample.calculate

            @wraps(original_calculate)
            def patched_calculate(self, texts, model, **kwargs):
                result = original_calculate(self, texts, model, **kwargs)
                if "sample_texts" in result:
                    result["sample_texts"] = [
                        [extract_final_answer_func(t) for t in texts_list]
                        for texts_list in result["sample_texts"]
                    ]
                return result

            sample.Sample.calculate = patched_calculate
            log.info("Patched Sample.calculate")
    except Exception as e:
        log.warning(f"Failed to patch Sample: {e}")


def apply_patch(extract_final_answer_func, model=None):
    """
    Apply all tokenizer decode patches.

    Args:
        extract_final_answer_func: A function that takes a string and returns
                                   the extracted final answer.
        model: Optional WhiteboxModel instance to patch. If None, patches the last created model.
    """
    log.info("Applying tokenizer decode patches...")

    # Patch the model's tokenizer if provided
    if model is not None and hasattr(model, "tokenizer"):
        original_decode = model.tokenizer.decode

        def patched_decode(token_ids, *args, **kwargs):
            result = original_decode(token_ids, *args, **kwargs)
            processed = extract_final_answer_func(result)
            if processed != result:
                log.debug(f"Applied extract_final_answer: '{result}' -> '{processed}'")
            return processed

        model.tokenizer.decode = patched_decode
        log.info(f"Successfully patched tokenizer.decode for model: {model.model_path}")
    else:
        log.warning("No model provided, skipping tokenizer.decode patch")

    patch_stat_calculators(extract_final_answer_func)
    log.info("Tokenizer decode patches applied successfully")


# Example usage:
# from your_module import extract_final_answer
# apply_patch(extract_final_answer)
