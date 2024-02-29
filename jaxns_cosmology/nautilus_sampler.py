import os
import time

import jax
import numpy as np
from bilby.core.sampler.base_sampler import NestedSampler, signal_wrapper
from jax import tree_map
from jaxns import resample
from nautilus import Prior, Sampler


class Nautilus(NestedSampler):
    default_kwargs = dict(
        use_nautilus_defaults=True,
        n_dim=None,
        n_live=2000,
        n_update=None,
        enlarge_per_dim=1.1,
        n_points_min=None,
        split_threshold=100,
        n_networks=4,
        neural_network_kwargs=dict(),
        prior_args=[],
        prior_kwargs=dict(),
        likelihood_args=[],
        likelihood_kwargs=dict(),
        n_batch=100,
        n_like_new_bound=None,
        vectorized=False,
        pass_dict=None,
        pool=None,
        n_jobs=None,
        seed=None,
        blobs_dtype=None,
        filepath=None,
        resume=True,
        discard_exploration=True
    )

    def __init__(
            self,
            likelihood,
            priors,
            outdir="outdir",
            label="label",
            plot=False,
            exit_code=77,
            skip_import_verification=False,
            temporary_directory=True,
            **kwargs
    ):
        super(Nautilus, self).__init__(
            likelihood=likelihood,
            priors=priors,
            outdir=outdir,
            label=label,
            plot=plot,
            exit_code=exit_code,
            **kwargs
        )

    def tuple_prod(self, t):
        """
        Product of shape tuple
        Args:
           t: tuple
        Returns:
           int
        """
        if len(t) == 0:
            return 1
        res = t[0]
        for a in t[1:]:
            res *= a
        return res

    def _get_vars(self, results, vars):
        if vars is None:
            vars = [k for k, v in results.samples.items()]
        vars = [v for v in vars if v in results.samples.keys()]
        vars = sorted(vars)
        return vars

    def _get_ndims(self, results, vars):
        ndims = int(sum([self.tuple_prod(v.shape[1:]) for k, v in results.samples.items() if (k in vars)]))
        return ndims

    @signal_wrapper
    def run_sampler(self):

        # self._verify_kwargs_against_default_kwargs()

        prior = Prior()
        _keys = list(self.priors)
        for key in _keys:
            prior.add_parameter(key, dist=(0., 1.))
        self._num_likelihood_evals = 0

        def likelihood(param_dict):
            self._num_likelihood_evals += 1
            self.likelihood.parameters = param_dict
            return self.likelihood.log_likelihood()

        discard_exploration = self.kwargs.pop('discard_exploration')

        use_nautilus_defaults = self.kwargs.pop('use_nautilus_defaults', False)

        # Create the nested sampler class. In this case without any tuning.
        sampler = Sampler(prior, likelihood, **self.kwargs)

        t0 = time.time()
        sampler.run(verbose=True, discard_exploration=discard_exploration)
        sampling_time = time.time() - t0

        samples, log_weights, log_L = sampler.posterior(return_as_dict=True, equal_weight=False)

        samples = resample(jax.random.PRNGKey(42), samples, log_weights,
                           S=2000, replace=True)

        try:
            import arviz as az
        except ImportError:
            raise RuntimeError("You must run `pip install arviz`")

        self.posterior = az.from_dict(posterior=tree_map(lambda x: np.asarray(x[None]), samples)).to_dataframe()

        # self.result.samples = tree_map(lambda x: np.asarray(x), samples)  #
        self.result.samples = self.posterior.drop(['chain', 'draw'], axis=1).to_numpy()

        # self._generate_result(ns_results)
        self.result.sampling_time = sampling_time

        self.result.log_evidence = sampler.evidence()

        self.result.log_likelihood_evaluations = log_L
        self.result.num_likelihood_evaluations = sampler.n_like
        self.result.ess = sampler.effective_sample_size()

        with open(os.path.join(self.outdir, f"{self.label}_summary.txt"), 'w') as file:
            file.write("ESS={}".format(int(self.result.ess)))
            file.write("\n")
            file.write("likelihood evals:{}".format(sampler.n_like))

        return self.result
