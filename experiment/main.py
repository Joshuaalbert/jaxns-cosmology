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
MAX_WALL_TIME_SECONDS = 3600.


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
    for c in [25, 50, 100]:
        for key, model in all_models().items():
            params = dict(
                nlive=model.U_ndims * c,
                maxcall=MAX_NUM_LIKELIHOOD_EVALUATIONS
            )
            yield key, model, params


def jaxns_models_and_parameters():
    for s in [5, 7, 10, 15]:
        for _c in [30, 50, 75, 100]:
            for key, model in all_models().items():
                params = dict(
                    max_samples=1e6,
                    difficult_model=True,
                    parameter_estimation=True,
                    c=model.U_ndims * _c,
                    s=s,
                    term_params=dict(
                        dlogZ=1e-4,
                        max_num_likelihood_evaluations=MAX_NUM_LIKELIHOOD_EVALUATIONS
                    ),
                    model=model
                )
                if key == 'MSSM7':
                    params['term_params']['dlogZ'] = 1e-6
                yield key, model, params


def nautilus_models_and_parameters():
    for c_ in [25, 50, 100]:
        for key, model in all_models().items():
            params = dict(
                n_live=model.U_ndims * c_
            )
            yield key, model, params


def nessai_models_and_parameters():
    for c_ in [25, 50, 100]:
        for key, model in all_models().items():
            params = dict(
                nlive=model.U_ndims * c_
            )
            yield key, model, params


def pymultinest_models_and_parameters():
    for c in [25, 50, 100]:
        for key, model in all_models().items():
            params = dict(
                n_live_points=model.U_ndims * c
            )
            yield key, model, params


def pypolychord_models_and_parameters():
    """
    Return all the models and their parameters

    Returns:
        A dictionary of models and their parameters and term params
    """
    for c_ in [25, 50, 100]:
        for key, model in all_models().items():
            params = dict(
                nlive=model.U_ndims * c_
            )
            yield key, model, params


def ultranest_models_and_parameters():
    for c in [25, 50, 100]:
        for key, model in all_models().items():
            params = dict(
                num_live_points=model.U_ndims * c,
                max_ncalls=MAX_NUM_LIKELIHOOD_EVALUATIONS
            )
            yield key, model, params


def main(sampler_name: str | None):
    experiments = [
        ('dynesty', dynesty_models_and_parameters()),
        ('pypolychord', pypolychord_models_and_parameters()),
        ('nautilus', nautilus_models_and_parameters()),
        # ('pymultinest', pymultinest_models_and_parameters()),
        ('jaxns', jaxns_models_and_parameters()),
        ('nessai', nessai_models_and_parameters()),
        ('ultranest', ultranest_models_and_parameters())
    ]
    for sampler, models_and_parameters in experiments:
        if sampler_name is not None:
            if sampler != sampler_name:
                continue
        experiment = Experiment(sampler=sampler, max_run_time=MAX_WALL_TIME_SECONDS)
        experiment.run(models_and_parameters)


if __name__ == '__main__':
    # Get sampler to run from sys args
    if len(sys.argv) != 2:
        main(None)
    else:
        sampler_name = sys.argv[1]
        main(sampler_name)
