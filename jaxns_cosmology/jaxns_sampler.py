import jax.numpy as jnp
import numpy as np
from bilby.core.sampler.base_sampler import NestedSampler
from jax import jit, random, tree_map
from jaxns import NestedSampler as OrigNestedSampler
from jaxns import PriorChain, UniformPrior
from jaxns.nested_sampler.utils import resample, summary


class Jaxns(NestedSampler):
    default_kwargs = dict(
        use_jaxns_defaults=True,
        sampler_name='slice',
        num_parallel_samplers=1,
        samples_per_step=None,
        # sampler_kwargs=None,
        max_samples=1e5,
        dynamic=False,
        verbose=True
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

        with PriorChain() as prior_chain:
            for key in self.priors:
                prior = UniformPrior(self.priors[key].name, low=self.priors[key].minimum, high=self.priors[key].maximum)

        if self.kwargs['use_jaxns_defaults']:
            # Create the nested sampler class. In this case without any tuning.
            ns = OrigNestedSampler(loglik_fn, prior_chain)
        else:
            jn_kwargs = self.kwargs.copy()
            jn_kwargs.pop('user_jaxns_defaults')
            ns = OrigNestedSampler(loglik_fn, prior_chain, **jn_kwargs)
        # We jit-compile
        ns = jit(ns)

        results = ns(random.PRNGKey(42424242),
                     termination_ess=None,
                     termination_evidence_uncert=None,
                     termination_live_evidence_frac=1e-4,
                     termination_max_num_steps=None,
                     termination_max_samples=None,
                     termination_max_num_likelihood_evaluations=None,
                     adaptive_evidence_patience=0,
                     adaptive_evidence_stopping_threshold=None,
                     G=0.,
                     num_live_points=None
                     )

        summary(results)
        # plot_diagnostics(results)
        # plot_cornerplot(results)

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

        samples = resample(random.PRNGKey(42), results.samples, results.log_dp_mean, S=int(results.ESS))

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

        #self.result.samples = tree_map(lambda x: np.asarray(x), samples)  #
        self.result.samples = self.posterior.drop(['chain', 'draw'], axis = 1).to_numpy()
