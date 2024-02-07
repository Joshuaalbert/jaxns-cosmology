import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp
from jaxns import Model, Prior

tfpd = tfp.distributions

#
# class Rosenbrock(bilby.Likelihood):
#     """
#     N-dimensional Rosenbrock as defined in
#     https://en.wikipedia.org/wiki/Rosenbrock_function
#
#     We take a = 1, b = 100
#
#     Args:
#         dimensionality: Number of input dimensions the function should take.
#
#     """
#
#     def __init__(self, dimensionality=2):
#         if dimensionality < 2:
#             raise Exception("""Dimensionality of Rosenbrock function has to
#                             be >=2.""")
#         self.dim = dimensionality
#
#         x_min = -5
#         x_max = 5
#
#         ranges = []
#         for i in range(self.dim):
#             ranges.append([x_min, x_max])
#         self.ranges = ranges
#
#         parameters = {"x{0}".format(i): 0 for i in range(self.dim)}
#
#         # Uniform priors are assumed
#         priors = bilby.core.prior.PriorDict()
#         priors.update({"x{0}".format(i): bilby.core.prior.Uniform(ranges[i][0], ranges[i][1], "x{0}".format(i)) for i in
#                        range(dim)})
#
#         self.priors = priors
#
#         super(Rosenbrock, self).__init__(parameters=parameters)
#
#     def log_likelihood(self):
#         x = np.array([self.parameters["x{0}".format(i)] for i in range(self.dim)])
#         y = 0
#         for i in range(self.dim - 1):
#             y += (100 * np.power(x[i + 1] - np.power(x[i], 2), 2) +
#                   np.power(1 - x[i], 2))
#         return -y

def build_rosenbrock_model(ndim: int) -> Model:
    """
    Builds the Rosenbrock model.

    Args:
        ndim: Number of input dimensions the function should take.

    Returns:
        model: The Rosenbrock model.
    """

    def prior_model():
        z = yield Prior(tfpd.Uniform(low=-5 * jnp.ones(ndim), high=5 * jnp.ones(ndim)), name='z')
        return z

    def log_likelihood(z):
        y = 0.
        for i in range(ndim - 1):
            y += (100. * jnp.power(z[i + 1] - jnp.power(z[i], 2), 2) + jnp.power(1 - z[i], 2))
        return -y

    model = Model(prior_model=prior_model, log_likelihood=log_likelihood)

    return model
