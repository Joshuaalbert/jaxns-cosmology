import os
import time

import jax
import numpy as np
import tensorflow_probability.substrates.jax as tfp
from bilby.core.sampler.base_sampler import NestedSampler
from jax import random, tree_map
from jaxns import summary, plot_diagnostics, DefaultNestedSampler, resample, plot_cornerplot, TerminationCondition

tfpd = tfp.distributions


class Jaxns(NestedSampler):
    default_kwargs = dict(
        use_jaxns_defaults=True,
        max_samples=1e7,
        num_live_points=None,
        s=None,
        k=None,
        c=None,
        num_parallel_workers=1,
        difficult_model=True,
        parameter_estimation=False,
        init_efficiency_threshold=0.1,
        verbose=False,
        term_params=dict(),
        model=None
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
        super(Jaxns, self).__init__(
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

    def run_sampler(self):

        # self._verify_kwargs_against_default_kwargs()

        if 'model' not in self.kwargs:
            raise ValueError(f"Model not found in kwargs. Please provide a model.")

        model = self.kwargs.pop('model')

        if 'term_params' in self.kwargs:
            term_cond = TerminationCondition(**self.kwargs.pop('term_params'))
        else:
            term_cond = None

        self.kwargs.pop('use_jaxns_defaults')

        jn_kwargs = self.kwargs.copy()
        ns = DefaultNestedSampler(
            model=model,
            **jn_kwargs
        )

        t0 = time.time()
        termination_reason, state = jax.jit(ns)(
            key=random.PRNGKey(42424242),
            term_cond=term_cond
        )
        sampling_time = time.time() - t0

        ns_results = ns.to_results(state=state, termination_reason=termination_reason, trim=True)

        os.makedirs(self.outdir, exist_ok=True)
        summary(ns_results, f_obj=os.path.join(self.outdir, f"{self.label}_summary.txt"))

        plot_diagnostics(ns_results, save_name=os.path.join(self.outdir, f"{self.label}_diagnostics.png"))
        plot_cornerplot(ns_results, save_name=os.path.join(self.outdir, f"{self.label}_cornerplot.png"))

        self._generate_result(ns_results, model)
        self.result.sampling_time = sampling_time

        return self.result

    def _generate_result(self, ns_results, model, vars=None):

        try:
            import arviz as az
        except ImportError:
            raise RuntimeError("You must run `pip install arviz`")

        self.result.log_evidence = np.asarray(ns_results.log_Z_mean)
        self.result.log_evidence_err = np.asarray(ns_results.log_Z_uncert)

        samples = resample(random.PRNGKey(42), ns_results.samples, ns_results.log_dp_mean,
                           S=max(model.U_ndims * 1000, int(ns_results.ESS)),
                           replace=True)

        self.result.num_likelihood_evaluations = np.asarray(ns_results.total_num_likelihood_evaluations)

        self.posterior = az.from_dict(posterior=tree_map(lambda x: np.asarray(x[None]), samples)).to_dataframe()

        # self.result.samples = tree_map(lambda x: np.asarray(x), samples)  #
        self.result.samples = self.posterior.drop(['chain', 'draw'], axis=1).to_numpy()
