import jax
import jax.numpy as jnp
import tensorflow as tf
import tensorflow_probability.substrates.jax as tfp
import tf2jax
from jaxns import Model, Prior

from jaxns_cosmology.models.ml_functions.ml_models import MLModels

tfpd = tfp.distributions


# class CMB(MLFunction, bilby.Likelihood):
#
#     def __init__(self):
#         self.modelname = 'lcdm'
#         self.dim = 6
#         self.x_mean = np.array([
#
#             0.95985349, 1.04158987, 0.02235954, 0.11994063, 0.05296935,
#             3.06873361], np.float64)
#         self.x_stdev = np.array([
#             0.00836984, 0.00168727, 0.00043045, 0.00470686, 0.00899083,
#             0.02362278], np.float64)
#
#         self.y_mean = 381.7929565376543
#         self.y_stdev = 1133.7707883293974
#
#         # Limit sampling between these hard borders, in order to ALWAYS remain
#         # within the training box.
#         ranges = []
#
#         # x_min = [0.92, 1.037, 0.02, 0.1, 0.0100002, 2.98
#         # ]
#         # x_max = [0.999999, 1.05, 0.024, 0.14, 0.097, 3.16
#         # ]
#
#         x_min = [0.954, 1.037, 0.02, 0.1, 0.05, 3.05
#                  ]
#         x_max = [0.966, 1.042, 0.0226, 0.14, 0.097, 3.16
#                  ]
#
#         for i in range(len(x_min)):
#             ranges.append([x_min[i], x_max[i]])
#         self.ranges = ranges
#
#         """
#         parameters = {'omega_b':0, 'omega_cdm':0, 'theta_s':0, 'ln[A_s]':0, 'n_s':0, 'tau_r':0}
#         labels = np.array(['$\\omega_b$', '$\\omega_{cdm}$', '$\\theta_s$', '$ln[A_s]$', '$n_s$', '$\\tau_r$'])
#         self.names = np.array(['omega_b', 'omega_cdm', 'theta_s', 'ln[A_s]', 'n_s', 'tau_r'])
#         """
#
#         parameters = {'n_s': 0, 'theta_s': 0, 'omega_b': 0, 'omega_cdm': 0, 'tau_r': 0, 'ln[A_s]': 0}
#         labels = np.array(['$n_s$', '$\\theta_s$', '$\\omega_b$', '$\\omega_{cdm}$', '$\\tau_r$', '$ln[A_s]$'])
#         self.names = np.array(['n_s', 'theta_s', 'omega_b', 'omega_cdm', 'tau_r', 'ln[A_s]'])
#
#         # Uniform priors are assumed
#         priors = dict()
#         i = 0
#         for key in parameters:
#             priors[key] = bilby.core.prior.Uniform(ranges[i][0], ranges[i][1], labels[i])
#             i += 1
#
#         self.priors = priors
#         super(CMB, self).__init__(parameters=parameters)
#
#     def log_likelihood(self):
#         if self.model is None:
#             self._load_model()
#         x = np.array([self.parameters[self.names[i]] for i in range(self.dim)])
#         cut = False
#         # x[0] -= 0.002
#         # x[1] -= 0.0005
#         # x[3] -= 0.001
#         x[4] += 0.01
#         x[5] += 0.02
#         if x[2] > 0.0224:
#             x[2] = 0.0446 - 1. * x[2]
#             # cut = True
#         if len(x.shape) == 1:
#             x = x.reshape(1, -1)
#         x = self._normalise(x, self.x_mean, self.x_stdev)
#         y = self.model.predict(x)
#         y = self._unnormalise(y, self.y_mean, self.y_stdev)
#         # if cut:
#         #  return -(y[0][0] - 0.28*y[0][0])
#         return -y[0][0]

def build_CMB_model() -> Model:
    """
    Builds the CMB model.

    Returns:
        model: The CMB model.
    """

    # Load hdf5 model, and use tf2jax to convert it to a JAX function
    # Use it inside the log_likelihood function

    def prior_model():
        x = yield Prior(tfpd.Uniform(low=jnp.array([0.954, 1.037, 0.02, 0.1, 0.05, 3.05]),
                                     high=jnp.array([0.966, 1.042, 0.0226, 0.14, 0.097, 3.16])), name='x')
        return x

    def log_likelihood(x: jnp.ndarray):
        ml_models = MLModels()

        # Load SavedModel
        restored = tf.keras.models.load_model(ml_models.get_file('lcdm'))

        @tf.function
        def predict(x):
            return restored(x, training=False)

        # Convert to JAX function
        jax_func, jax_params = tf2jax.convert(predict, jnp.zeros((1, 6), jnp.float32))

        x_mean = jnp.array([0.95985349, 1.04158987, 0.02235954, 0.11994063, 0.05296935, 3.06873361], jnp.float32)
        x_stdev = jnp.array([0.00836984, 0.00168727, 0.00043045, 0.00470686, 0.00899083, 0.02362278], jnp.float32)
        y_mean = 381.7929565376543
        y_stdev = 1133.7707883293974

        #         cut = False
        #         # x[0] -= 0.002
        #         # x[1] -= 0.0005
        #         # x[3] -= 0.001
        #         x[4] += 0.01
        #         x[5] += 0.02
        #         if x[2] > 0.0224:
        #             x[2] = 0.0446 - 1. * x[2]
        #             # cut = True
        #         if len(x.shape) == 1:
        #             x = x.reshape(1, -1)
        #         x = self._normalise(x, self.x_mean, self.x_stdev)
        #         y = self.model.predict(x)
        #         y = self._unnormalise(y, self.y_mean, self.y_stdev)
        #         # if cut:
        #         #  return -(y[0][0] - 0.28*y[0][0])
        #         return -y[0][0]

        # mod x
        # x = x.at[0].add(-0.002)
        # x = x.at[1].add(-0.0005)
        # x = x.at[3].add(-0.001)
        x = x.at[4].add(0.01)
        x = x.at[5].add(0.02)
        x = jnp.where(x[2] > 0.0224, x.at[2].set(0.0446 - 1. * x[2]), x)
        # cut = (x[2] > 0.0224)
        x = (x - x_mean) / x_stdev
        x = jnp.reshape(x, (1, -1))
        y, _ = jax_func(jax_params, x, rng=jax.random.PRNGKey(42))
        y = (y * y_stdev) + y_mean
        # y = jnp.where(cut, -(y - 0.28 * y), -y)
        return -y.reshape(())

    model = Model(prior_model=prior_model, log_likelihood=log_likelihood)

    return model
