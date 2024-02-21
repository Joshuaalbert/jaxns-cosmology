import glob
import os
from concurrent import futures

import bilby
import jax.numpy as jnp

from jaxns_cosmology import install_jaxns_sampler, install_nautilus_sampler

# Set export CUDA_VISIBLE_DEVICES=""
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import time
from typing import Dict, Any, Tuple, Generator, TypeVar, Callable

from jaxns import Model
from pydantic import BaseModel

from jaxns_cosmology.models.convert import convert_model

install_jaxns_sampler()
install_nautilus_sampler()


class ExperimentResult(BaseModel):
    model_name: str
    params: Dict[str, Any]
    likelihood_evals: int


T = TypeVar('T')


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

class TooManyLikelihoodEvals(Exception):
    pass

class TooLong(Exception):
    pass

class Experiment:
    def __init__(self, sampler: str, max_run_time: float, max_likelihood_evals: int):
        self.sampler = sampler
        self.max_run_time = max_run_time
        self.max_likelihood_evals = max_likelihood_evals

    def run_model(self, model_name: str, model, params):

        kwargs = params.copy()

        forward = convert_model(model)

        class Likelihood(bilby.Likelihood):
            def __init__(self,
                         max_run_time=self.max_run_time,
                         max_likelihood_evals=self.max_likelihood_evals
                         ):
                """A very simple Gaussian likelihood"""
                super().__init__(parameters={f"x{i}": None for i in range(model.U_ndims)})
                self.__num_likelihood_evals = 0
                self.__max_run_time = max_run_time
                self.__start_time = time.time()
                self.__max_likelihood_evals = max_likelihood_evals

            def log_likelihood(self):
                """Log-likelihood."""
                u = jnp.asarray([self.parameters[f"x{i}"] for i in range(model.U_ndims)])
                self.__num_likelihood_evals += 1
                if self.__num_likelihood_evals > self.__max_likelihood_evals:
                    raise TooManyLikelihoodEvals(f"Exceeded maximum likelihood evaluations of {self.__max_likelihood_evals}")
                if time.time() - self.__start_time > self.__max_run_time:
                    raise TooLong(f"Exceeded maximum run time of {self.__max_run_time} seconds")
                return float(forward(u))

        priors = {
            f"x{i}": bilby.core.prior.Uniform(0., 1., f"x{i}")
            for i in range(model.U_ndims)
        }

        # get tmp dir for output
        outdir = './'
        label = self.sampler
        likelihood = Likelihood()

        result = bilby.run_sampler(
            likelihood=likelihood,
            priors=priors,
            outdir=outdir,
            label=label,
            plot=True,
            sampler=self.sampler,
            injection_parameters={f"x{i}": 0.0 for i in range(model.U_ndims)},
            likelihood_benchmark=True,
            seed=1234,
            **kwargs
        )

        # First remove params that are not serialisable
        for k in list(params.keys()):
            if not isinstance(params[k], (int, float, str, bool, list, dict)):
                del params[k]

        experiment_result = ExperimentResult(
            model_name=model_name,
            params=params,
            likelihood_evals=result.num_likelihood_evaluations
        )
        with open(f"{model_name}_experiment_result.json", "w") as f:
            f.write(experiment_result.json(indent=2))

    def run(self, model_and_param_gen: Generator[Tuple[str, Model, Dict[str, Any]], None, None]):
        main_dir = os.path.abspath(os.getcwd())
        for model_name, model, params in model_and_param_gen:
            os.chdir(main_dir)
            run_dir_glob = os.path.join(main_dir, self.sampler, f"{model_name}_run_*")

            past_runs = glob.glob(run_dir_glob)
            # See if it's run before
            same = False
            for past_run in past_runs:
                if not os.path.exists(os.path.join(past_run, f"{model_name}_experiment_result.json")):
                    continue
                with open(os.path.join(past_run, f"{model_name}_experiment_result.json"), 'r') as f:
                    result = ExperimentResult.parse_raw(f.read())
                    same = all(params.get(k, None) == v for k, v in result.params.items())
                if same:
                    break
            if same:
                continue

            new_run = os.path.join(main_dir, self.sampler, f"{model_name}_run_{len(past_runs):03d}")

            # create folder and cd there
            os.makedirs(new_run, exist_ok=True)
            os.chdir(new_run)

            print(f"Running model {model_name} with params {params} using {self.sampler}")

            t0 = time.time()
            try:
                timeout_run(self.run_model, model_name, model, params, timeout=self.max_run_time)
            except futures.TimeoutError:
                print(f"Model {model_name} {params} timed out after {self.max_run_time} seconds.")
                continue
            except KeyboardInterrupt:
                print(f"Model {model_name} {params} was interrupted.")
                continue
            except InterruptedError:
                print(f"Model {model_name} {params} was interrupted.")
                continue
            except TooManyLikelihoodEvals:
                print(f"Model {model_name} {params} exceeded maximum likelihood evaluations of {self.max_likelihood_evals}")
                continue
            except TooLong:
                print(f"Model {model_name} {params} exceeded maximum run time of {self.max_run_time} seconds.")
                continue

            t = time.time() - t0

            print(f"Model {model_name} took {t} seconds to run")

        results = []
        for filename in glob.glob(os.path.join(main_dir, self.sampler, f"{self.sampler}_experiment_result.json")):
            with open(filename, "r") as f:
                results.append((filename, ExperimentResult.parse_raw(f.read())))

        # sort by likelihood evals and print
        results = sorted(results, key=lambda x: x[1].likelihood_evals)
        for filename, result in results:
            print(f"{filename}: {result.likelihood_evals} likelihood evals")

        os.chdir(main_dir)
