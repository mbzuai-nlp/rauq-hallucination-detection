from lm_polygraph.stat_calculators.sample import (
    SamplingGenerationCalculator,
)


def load_stat_calculator(config, builder):
    return SamplingGenerationCalculator(config.samples_n)
