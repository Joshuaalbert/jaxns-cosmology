from jax import config

config.update("jax_enable_x64", True)

import matplotlib

matplotlib.use('Agg')

import os
import sys

# Set export CUDA_VISIBLE_DEVICES=""
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from typing import TypeVar

from jaxns_cosmology.models import all_models
from jaxns_cosmology.experiment import Experiment

T = TypeVar('T')


def jaxns_models_and_parameters(model_name):
    for key, model in all_models().items():
        if key != model_name:
            continue
        params = dict(
            max_samples=1e7,
            k=0,
            c=model.U_ndims * 2000,
            s=model.U_ndims * 10,
            term_params=dict(
                dlogZ=1e-6
            ),
            model=model,
            difficult_model=True,
            verbose=True
        )
        yield key, model, params


def main(model_name: str):
    experiment = Experiment(
        sampler='jaxns',
        max_run_time=86400.,  # 24 hours
        max_likelihood_evals=int(1e15)  # Will never reach this
    )
    experiment.run(jaxns_models_and_parameters(model_name))


if __name__ == '__main__':
    # Get sampler to run from sys args
    if len(sys.argv) != 2:
        raise ValueError("Please provide a model name")
    else:
        model_name = sys.argv[1]
        main(model_name)
