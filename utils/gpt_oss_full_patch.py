"""
Patch to filter attention for GPT-OSS models to only return full attention layers.

GPT-OSS models use a mix of sliding attention and full attention layers. This patch
ensures that only attention from full attention layers is returned when output_attentions=True.
"""

import logging
import torch

log = logging.getLogger("lm_polygraph.gpt_oss_full_patch")


def apply_patch(model, max_new_tokens: int = 0):
    """
    Patch the model to filter attention for full attention layers only.

    Args:
        model: WhiteboxModel instance containing the GPT-OSS model
        max_new_tokens: Maximum number of tokens to generate. Patch is only applied if > 128.
    """
    # Only apply patch if max_new_tokens > 128
    if max_new_tokens <= 128:
        log.info(
            f"GPT-OSS full attention patch skipped: max_new_tokens={max_new_tokens} <= 128"
        )
        return

    if not hasattr(model.model, "config"):
        log.warning("Model does not have config, skipping GPT-OSS full attention patch")
        return

    config = model.model.config
    if not hasattr(config, "layer_types"):
        log.warning(
            "Model config does not have layer_types, skipping GPT-OSS full attention patch"
        )
        return

    # Identify full attention layers (those not using sliding_attention)
    layer_types = config.layer_types
    full_attn_layer_ids = [
        i for i, t in enumerate(layer_types) if t != "sliding_attention"
    ]

    log.info(f"GPT-OSS layer_types: {layer_types}")
    log.info(f"Full attention layer IDs: {full_attn_layer_ids}")

    if not full_attn_layer_ids:
        log.warning("No full attention layers found, skipping patch")
        return

    # Store the original forward method
    original_forward = model.model.forward

    def patched_forward(*args, **kwargs):
        # Call original forward
        outputs = original_forward(*args, **kwargs)

        # Only filter if attentions are being output
        if outputs.attentions is not None:
            # Filter attentions to only include full attention layers
            filtered_attentions = tuple(
                outputs.attentions[i] for i in full_attn_layer_ids
            )

            # Create a new output tuple with filtered attentions
            if isinstance(outputs, tuple):
                # Replace the attentions in the tuple
                # attentions is typically the second-to-last element or last element
                new_outputs = list(outputs)
                new_outputs[-2] = filtered_attentions  # attentions is typically at -2
                outputs = type(outputs)(*new_outputs)
            elif hasattr(outputs, "attentions"):
                outputs.attentions = filtered_attentions

        return outputs

    # Patch the forward method
    model.model.forward = patched_forward

    # Store patch metadata for reference
    if not hasattr(model, "_gpt_oss_patch_info"):
        model._gpt_oss_patch_info = {}

    model._gpt_oss_patch_info["full_attn_layer_ids"] = full_attn_layer_ids
    model._gpt_oss_patch_info["original_forward"] = original_forward

    log.info(
        f"Successfully patched GPT-OSS model for full attention filtering. "
        f"Returning attention for {len(full_attn_layer_ids)} layers only."
    )


def map_layer_indices(
    original_layers: list[int], full_attn_layer_ids: list[int]
) -> list[int]:
    """
    Map original layer indices to filtered layer indices after patch is applied.

    Args:
        original_layers: List of original layer indices (e.g., [14, 15, 16, 17, 18])
        full_attn_layer_ids: List of full attention layer IDs (e.g., [1, 3, 5, ..., 23])

    Returns:
        List of mapped layer indices in the filtered output (0-indexed)
    """
    mapped = []
    for layer_id in original_layers:
        if layer_id in full_attn_layer_ids:
            # Find the index in the filtered output
            mapped_index = full_attn_layer_ids.index(layer_id)
            mapped.append(mapped_index)
    return mapped
