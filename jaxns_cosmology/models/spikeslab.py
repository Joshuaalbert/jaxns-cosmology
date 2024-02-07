import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp
from jaxns import Model, Prior

tfpd = tfp.distributions


# class SpikeSlab(bilby.Likelihood):
#
#     def __init__(self, dimensionality=2):
#         if dimensionality < 2:
#             raise Exception("""Dimensionality of SpikeSlab function has to
#                             be >=2.""")
#
#         self.dim = dimensionality
#
#         x_min = -4
#         x_max = 8.
#
#         ranges = []
#         for i in range(self.dim):
#             ranges.append([x_min, x_max])
#         self.ranges = ranges
#
#         parameters = {"x{0}".format(i): 0 for i in range(self.dim)}
#
#         #Uniform priors are assumed
#         priors = bilby.core.prior.PriorDict()
#         priors.update({"x{0}".format(i): bilby.core.prior.Uniform(ranges[i][0], ranges[i][1], "x{0}".format(i)) for i in range(dim)})
#
#         self.priors = priors
#
#         super(SpikeSlab, self).__init__(parameters=parameters)
#
#     def log_likelihood(self):
#         x = np.array([self.parameters["x{0}".format(i)] for i in range(self.dim)])
#         mean_1 = [6,6]
#         mean_2 = [2.5,2.5]
#         for i in range(self.dim-2):
#             mean_1.append(0)
#             mean_2.append(0)
#         cov_1 = 0.08*np.identity(self.dim)
#         cov_2 = 0.8*np.identity(self.dim)
#         gauss_1 = stats.multivariate_normal.pdf(x,mean = mean_1,cov = cov_1)
#         gauss_2 = stats.multivariate_normal.pdf(x,mean = mean_2,cov = cov_2)
#         y = np.log(gauss_1 + gauss_2)
#         return y

def build_spikeslab_model(ndim: int) -> Model:
    """
    Builds the SpokeSlab model.

    Args:
        ndim: Number of input dimensions the function should take.

    Returns:
        model: The SpokeSlab model.
    """

    def prior_model():
        z = yield Prior(tfpd.Uniform(low=-4. * jnp.ones(ndim), high=8. * jnp.ones(ndim)), name='z')
        return z

    def log_likelihood(z):
        mean_1 = jnp.array([6., 6.])
        mean_2 = jnp.array([2.5, 2.5])
        for i in range(ndim - 2):
            mean_1 = jnp.append(mean_1, 0.)
            mean_2 = jnp.append(mean_2, 0.)
        cov_1 = 0.08 * jnp.eye(ndim)
        cov_2 = 0.8 * jnp.eye(ndim)
        gauss_1 = tfp.distributions.MultivariateNormalFullCovariance(loc=mean_1, covariance_matrix=cov_1).log_prob(z)
        gauss_2 = tfp.distributions.MultivariateNormalFullCovariance(loc=mean_2, covariance_matrix=cov_2).log_prob(z)
        y = jnp.logaddexp(gauss_1, gauss_2)
        return y

    model = Model(prior_model=prior_model, log_likelihood=log_likelihood)

    return model
