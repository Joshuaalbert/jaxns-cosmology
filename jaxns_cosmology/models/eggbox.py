import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp
from jaxns import Model, Prior

tfpd = tfp.distributions


# class EggBox(bilby.Likelihood):
#     """
#     N-dimensional EggBox as defined in arXiv:0809.3437.
#     Testfunction as defined in arXiv:0809.3437
#
#     Args:
#         dimensionality: Number of input dimensions the function should take.
#
#     """
#
#     def __init__(self, dimensionality=2):
#         if dimensionality < 2:
#             raise Exception("""Dimensionality of EggBox function has to
#                             be >=2.""")
#         self.dim = dimensionality
#         self.tmax = 5.0 * np.pi
#
#         x_min = 0.
#         x_max = 10.*np.pi
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
#         super(EggBox, self).__init__(parameters=parameters)
#
#     def log_likelihood(self):
#         x = np.array([self.parameters["x{0}".format(i)] for i in range(self.dim)])
#         y = 1
#
#         for i in range(self.dim):
#             y *= math.cos(x[i]/2)
#         y = math.pow(2. + y, 5)
#         return y

def build_eggbox_model(ndim: int):
    """
    Builds the eggbox model.

    Args:
        ndim:  The number of dimensions of the eggbox function.

    Returns:
        forward: A function that takes a flat array of i.i.d. samples of U[0,1] and returns the eggbox function.
    """
    def prior_model():
        z = yield Prior(tfpd.Uniform(low=jnp.zeros(ndim), high=10. * jnp.pi * jnp.ones(ndim)), name='z')
        return z

    def log_likelihood(z):
        y = 1
        for i in range(ndim):
            y *= jnp.cos(z[i] / 2)
        y = jnp.power(2. + y, 5)
        return y

    model = Model(prior_model=prior_model,
                  log_likelihood=log_likelihood)
    forward = jax.jit(lambda U: model.forward(U=U, allow_nan=True))
    return forward
