import jax
import jax.numpy as jnp
import tensorflow as tf
import tensorflow_probability.substrates.jax as tfp
import tf2jax
from jaxns import Model, Prior

from jaxns_cosmology.models.ml_functions.ml_models import MLModels

tfpd = tfp.distributions


def build_CMB_model() -> Model:
    """
    Builds the CMB model.

    Returns:
        model: The CMB model.
    """

    # Load hdf5 model, and use tf2jax to convert it to a JAX function
    # Use it inside the log_likelihood function

    def prior_model():
        x_min = jnp.array([0.954, 1.037, 0.02, 0.1, 0.05, 3.05])
        x_max = jnp.array([0.966, 1.042, 0.0226, 0.14, 0.097, 3.16])
        x = yield Prior(tfpd.Uniform(low=x_min, high=x_max), name='x')
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
        x = jnp.reshape(x, (1, -1)).astype(jnp.float32)
        y, _ = jax_func(jax_params, x, rng=jax.random.PRNGKey(42))
        y = (y * y_stdev) + y_mean
        # y = jnp.where(cut, -(y - 0.28 * y), -y)
        return -y.reshape(())

    model = Model(prior_model=prior_model, log_likelihood=log_likelihood)

    return model
