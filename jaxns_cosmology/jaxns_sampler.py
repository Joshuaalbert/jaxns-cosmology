import os
from functools import cached_property

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax as tfp
from bilby.core.sampler.base_sampler import NestedSampler
from jax import random, tree_map

from jaxns import sample_evidence, summary, plot_diagnostics, DefaultNestedSampler, resample, Prior, Model, \
    plot_cornerplot, TerminationCondition

tfpd = tfp.distributions


class Jaxns(NestedSampler):
    default_kwargs = dict(
        use_jaxns_defaults=True,
        max_samples=1e6
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

    @cached_property
    def model(self) -> Model:
        param_names = []
        for key in self.priors:
            param_names.append(self.priors[key].name)

        # Jaxns requires loglikelihood function to have explicit signatures.
        local_dict = {}
        loglik_fn_def = """def loglik_fn({}):\n
                \tparams = [{}]\n
                \treturn self.log_likelihood(params)
                """.format(
            ", ".join([f"{name}" for name in param_names]),
            ", ".join([f"{name}" for name in param_names]),
        )

        exec(loglik_fn_def, locals(), local_dict)
        loglik_fn = local_dict["loglik_fn"]

        def prior_model():
            params = []
            for key in self.priors:
                param = yield Prior(tfpd.Uniform(low=self.priors[key].minimum, high=self.priors[key].maximum),
                                    name=self.priors[key].name)
                params.append(param)
            return tuple(params)

        model = Model(prior_model=prior_model, log_likelihood=loglik_fn)

        return model

    def run_sampler(self):

        # self._verify_kwargs_against_default_kwargs()

        if 'live_evidence_frac' in self.kwargs:
            term_cond = TerminationCondition(live_evidence_frac=self.kwargs.pop('live_evidence_frac'))
        else:
            term_cond = TerminationCondition()

        if self.kwargs['use_jaxns_defaults']:
            # Create the nested sampler class. In this case without any tuning.
            ns = DefaultNestedSampler(
                model=self.model,
                max_samples=1e6,
                parameter_estimation=True,
                difficult_model=True
            )
        else:
            jn_kwargs = self.kwargs.copy()
            jn_kwargs.pop('use_jaxns_defaults')
            ns = DefaultNestedSampler(
                model=self.model,
                max_samples=1e6,
                parameter_estimation=True,
                difficult_model=True,
                **jn_kwargs
            )

        termination_reason, state = jax.jit(ns)(
            key=random.PRNGKey(42424242),
            term_cond=term_cond
        )

        results = ns.to_results(state=state, termination_reason=termination_reason)

        log_Z = sample_evidence(
            key=state.key,
            num_live_points_per_sample=results.num_live_points_per_sample,
            log_L_samples=results.log_L_samples,
            S=1000
        )
        log_Z_mean = jnp.nanmean(log_Z)
        log_Z_std = jnp.nanstd(log_Z)
        results = results._replace(
            log_Z_mean=jnp.where(jnp.isnan(log_Z_mean), results.log_Z_mean, log_Z_mean),
            log_Z_uncert=jnp.where(jnp.isnan(log_Z_std), results.log_Z_uncert, log_Z_std)
        )
        os.makedirs(self.outdir, exist_ok=True)
        summary(results, f_obj=os.path.join(self.outdir, f"{self.label}_summary.txt"))
        plot_diagnostics(results, save_name=os.path.join(self.outdir, f"{self.label}_diagnostics.png"))
        plot_cornerplot(results)

        self._generate_result(results)
        # self.result.sampling_time = self.sampling_time

        return self.result

    def _generate_result(self, results, vars=None):

        try:
            import arviz as az
        except ImportError:
            raise RuntimeError("You must run `pip install arviz`")

        self.result.log_evidence = np.asarray(results.log_Z_mean)
        self.result.log_evidence_err = np.asarray(results.log_Z_uncert)

        rkey0 = random.PRNGKey(123496)
        vars = self._get_vars(results, vars)
        ndims = self._get_ndims(results, vars)

        nsamples = results.total_num_samples
        max_like_idx = jnp.argmax(results.log_L_samples[:nsamples])
        map_idx = jnp.argmax(results.log_dp_mean)
        log_L = results.log_L_samples[:nsamples]
        log_p = results.log_dp_mean[:nsamples]
        log_p = jnp.exp(log_p)
        # log_L = jnp.where(jnp.isfinite(log_L), log_L, -jnp.inf)
        samples = resample(random.PRNGKey(42), results.samples, results.log_dp_mean,
                           S=max(self.model.U_ndims * 1000, int(results.ESS)))

        # l = []
        # lkeys = ['weights', 'log_likelihood']
        # l.append(log_p.reshape(nsamples, -1))
        # l.append(log_L.reshape(nsamples, -1))
        # for key in vars:
        #     samples = results.samples[key][:nsamples, ...].reshape((nsamples, -1))
        #     l.append(samples)
        #     lkeys.append(key)
        # samples_row = np.asarray(jnp.stack(l).squeeze(-1).T)
        #
        # nested_samples = DataFrame(samples_row, columns=lkeys)
        # self.result.nested_samples = nested_samples

        self.result.num_likelihood_evaluations = np.asarray(results.total_num_likelihood_evaluations)
        # self.result.log_likelihood_evaluations = results.log_efficiency

        self.posterior = az.from_dict(posterior=tree_map(lambda x: np.asarray(x[None]), samples)).to_dataframe()

        # self.result.samples = tree_map(lambda x: np.asarray(x), samples)  #
        self.result.samples = self.posterior.drop(['chain', 'draw'], axis=1).to_numpy()
