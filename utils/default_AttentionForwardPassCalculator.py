from attention_forward_pass import (
    AttentionForwardPassCalculator,
)


def load_stat_calculator(config, builder):
    return AttentionForwardPassCalculator(max_input_length=config.max_input_length)
