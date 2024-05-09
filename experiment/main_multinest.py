from jax import config

config.update("jax_enable_x64", True)

import matplotlib

matplotlib.use('Agg')

import os
import sys

# Set export CUDA_VISIBLE_DEVICES=""
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from concurrent import futures
from typing import TypeVar, Callable

from jaxns_cosmology.models import all_models
from jaxns_cosmology.experiment import Experiment

T = TypeVar('T')

MAX_NUM_LIKELIHOOD_EVALUATIONS = int(50e6)
MAX_WALL_TIME_SECONDS = 2 * 3600.


def timeout_run(func: Callable[..., T], *args, timeout: float | None = None) -> T:
    """
    Run a function with a timeout

    Args:
        func: the function to run
        *args: the arguments to pass to the function
        timeout: the timeout in seconds

    Returns:
        the result of the function

    Raises:
        TimeoutError: if the function takes longer than the timeout
    """

    # use threading so we share process

    def run_with_timeout():
        return func(*args)

    with futures.ThreadPoolExecutor() as pool:
        future = pool.submit(run_with_timeout)
        return future.result(timeout=timeout)


def pymultinest_models_and_parameters(model_name):
    for key, model in all_models().items():
        if key == 'rastrigin':
            for c in [25, 50, 100, 200, 500, 1000]:

                if key is not None:
                    if key != model_name:
                        continue
                params = dict(
                    n_live_points=model.U_ndims * c
                )
                yield key, model, params

        else:
            for c in [25, 50, 100]:

                if key is not None:
                    if key != model_name:
                        continue
                params = dict(
                    n_live_points=model.U_ndims * c
                )
                yield key, model, params


def main(model_name: str | None):
    experiments = [
        ('pymultinest', pymultinest_models_and_parameters(model_name))
    ]
    # The problem is that multinest doesn't allow exceptions to terminate the implementation, so we need to run each
    # experiment separately and manually monitor for experiment resources.
    for sampler, models_and_parameters in experiments:
        experiment = Experiment(
            sampler=sampler,
            max_run_time=86400.,  # 24 hours
            max_likelihood_evals=int(1e15)  # Will never reach this
        )
        experiment.run(models_and_parameters)


if __name__ == '__main__':
    # Get sampler to run from sys args
    if len(sys.argv) != 2:
        main(None)
    else:
        model_name = sys.argv[1]
        main(model_name)
