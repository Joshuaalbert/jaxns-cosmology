import jax
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


def dynesty_models_and_parameters():
    """
    Return all the models and their parameters

    Returns:
        A dictionary of models and their parameters and term params
    """
    for key, model in all_models().items():
        if key == 'rastrigin':
            for c in [25, 50, 100, 200, 500, 1000]:
                params = dict(
                    nlive=model.U_ndims * c,
                    # maxcall=MAX_NUM_LIKELIHOOD_EVALUATIONS
                )
                yield key, model, params
        else:
            for c_ in [25, 50, 100]:
                params = dict(
                    nlive=model.U_ndims * c_,
                    maxcall=MAX_NUM_LIKELIHOOD_EVALUATIONS
                )
                yield key, model, params


def jaxns_models_and_parameters():
    for key, model in all_models().items():
        if key == 'rastrigin':
            for s in [5, 10]:
                for c in [model.U_ndims * 25, model.U_ndims * 50, model.U_ndims * 100,
                          model.U_ndims * 200, model.U_ndims * 500, model.U_ndims * 1000]:
                    for k in [model.U_ndims // 2]:
                        params = dict(
                            max_samples=1e7,
                            k=k,
                            c=c,
                            s=s,
                            term_params=dict(
                                dlogZ=1e-4,
                                max_num_likelihood_evaluations=1e10
                            ),
                            model=model,
                            difficult_model=True
                        )
                        yield key, model, params
        else:
            for s in [5]:
                for c in [model.U_ndims * 25, model.U_ndims * 50, model.U_ndims * 100]:
                    for k in list(range(model.U_ndims // 2 + 1)):
                        params = dict(
                            max_samples=1e7,
                            k=k,
                            c=c,
                            s=s,
                            term_params=dict(
                                dlogZ=1e-4,
                                max_num_likelihood_evaluations=1e10
                            ),
                            model=model,
                            difficult_model=True
                        )
                        yield key, model, params


def nautilus_models_and_parameters():
    for key, model in all_models().items():
        if key == 'rastrigin':
            for c in [25, 50, 100, 200, 500, 1000]:
                params = dict(
                    n_live=model.U_ndims * c
                )
                yield key, model, params
        else:
            for c_ in [25, 50, 100]:
                params = dict(
                    n_live=model.U_ndims * c_
                )
                yield key, model, params


def nessai_models_and_parameters():
    for key, model in all_models().items():
        if key == 'rastrigin':
            for c in [25, 50, 100, 200, 500, 1000]:
                params = dict(
                    nlive=model.U_ndims * c
                )
                yield key, model, params
        else:
            for c_ in [25, 50, 100]:
                params = dict(
                    nlive=model.U_ndims * c_
                )
                yield key, model, params


def pypolychord_models_and_parameters():
    """
    Return all the models and their parameters

    Returns:
        A dictionary of models and their parameters and term params
    """
    for key, model in all_models().items():
        if key == 'rastrigin':
            for c in [25, 50, 100, 200, 500, 1000]:
                params = dict(
                    nlive=model.U_ndims * c
                )
                yield key, model, params
        else:
            for c_ in [25, 50, 100]:
                params = dict(
                    nlive=model.U_ndims * c_
                )
                yield key, model, params


def ultranest_models_and_parameters():
    for key, model in all_models().items():
        if key == 'rastrigin':
            for c in [25, 50, 100, 200, 500, 1000]:
                # Explore both static and reactive modes

                params = dict(
                    min_num_live_points=model.U_ndims * c,
                    max_num_improvement_loops=-1  # Makes it use reactive mode
                )
                yield key, model, params

                params = dict(
                    min_num_live_points=model.U_ndims * c,
                    max_num_improvement_loops=0  # Make it use the static mode
                )
                yield key, model, params

        else:
            for c in [25, 50, 100]:
                # Explore both static and reactive modes

                params = dict(
                    min_num_live_points=model.U_ndims * c,
                    max_ncalls=MAX_NUM_LIKELIHOOD_EVALUATIONS,
                    max_num_improvement_loops=-1  # Makes it use reactive mode
                )
                yield key, model, params

                params = dict(
                    min_num_live_points=model.U_ndims * c,
                    max_ncalls=MAX_NUM_LIKELIHOOD_EVALUATIONS,
                    max_num_improvement_loops=0  # Make it use the static mode
                )
                yield key, model, params


def main(sampler_name: str | None):
    experiments = [
        ('dynesty', dynesty_models_and_parameters()),
        ('pypolychord', pypolychord_models_and_parameters()),
        ('nautilus', nautilus_models_and_parameters()),
        ('jaxns', jaxns_models_and_parameters()),
        ('nessai', nessai_models_and_parameters()),
        ('ultranest', ultranest_models_and_parameters())
    ]
    for sampler, models_and_parameters in experiments:
        if sampler_name is not None:
            if sampler != sampler_name:
                continue
        experiment = Experiment(
            sampler=sampler,
            max_run_time=MAX_WALL_TIME_SECONDS,
            max_likelihood_evals=MAX_NUM_LIKELIHOOD_EVALUATIONS
        )
        experiment.run(models_and_parameters)
        jax.clear_caches()


if __name__ == '__main__':
    # Get sampler to run from sys args
    if len(sys.argv) != 2:
        main(None)
    else:
        sampler_name = sys.argv[1]
        main(sampler_name)
