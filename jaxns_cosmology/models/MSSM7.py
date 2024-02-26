import jax
import jax.numpy as jnp
import tensorflow as tf
import tensorflow_probability.substrates.jax as tfp
import tf2jax
from jaxns import Model, Prior

from jaxns_cosmology.models.ml_functions.ml_models import MLModels

tfpd = tfp.distributions


def build_MSSM7_model() -> Model:
    """
    Builds the MSSM7 model.


    Returns:
        model: The MSSM7 model.
    """

    # Load SavedModel, and use tf2jax to convert it to a JAX function
    # Use it inside the log_likelihood function

    def prior_model():
        x = yield Prior(tfpd.Uniform(low=jnp.zeros(12),
                                     high=jnp.ones(12)), name='x')
        return x

    def log_likelihood(x: jnp.ndarray):
        ml_models = MLModels()
        # Load SavedModel
        restored = tf.keras.models.load_model(ml_models.get_file('mssm7'))

        @tf.function
        def predict(x):
            return restored(x, training=False)

        x_min = jnp.array([-7.16775760e+03, 4.27547804e+05, -9.98192815e+07, -6.81824964e+07,
                           -9.99995488e+03, -9.99999903e+03, 3.00597043e+00, 1.71060011e+02,
                           1.16700013e-01, 2.00000156e-01, 1.90001455e+01, 3.10001673e+01], jnp.float64)
        x_max = jnp.array([7.18253463e+03, 9.99999857e+07, 4.56142832e+05, 9.99999734e+07,
                           9.99987623e+03, 9.99999881e+03, 6.99999394e+01, 1.75619963e+02,
                           1.20299997e-01, 7.99999435e-01, 6.69997800e+01, 8.49983345e+01], jnp.float64)
        x = x * (x_max - x_min) + x_min

        # Load SavedModel
        jax_func, jax_params = tf2jax.convert(predict, jnp.zeros((1, 12), jnp.float32))

        x_mean = jnp.array([-1.65550622e+02, 6.53242357e+07, -6.04267288e+06, 1.15227686e+07,
                            -8.96546390e+02, 1.20880748e+03, 3.65456629e+01, 1.73423279e+02,
                            1.18539912e-01, 4.00306869e-01, 4.31081695e+01, 5.80441328e+01], jnp.float64)
        x_stdev = jnp.array([3.13242671e+03, 2.64037878e+07, 4.32735828e+06, 2.22328069e+07,
                             2.33891832e+03, 6.20060930e+03, 1.40566829e+01, 3.83628710e-01,
                             2.51362291e-04, 4.59609868e-02, 3.09244370e+00, 3.24780776e+00], jnp.float64)
        y_mean = -262.5887645450105
        y_stdev = 7.461633956842537

        x = (x.astype(jnp.float64) - x_mean) / x_stdev
        x = jnp.reshape(x, (1, -1)).astype(jnp.float32)
        y, _ = jax_func(jax_params, x, rng=jax.random.PRNGKey(42))
        y = (y * y_stdev) + y_mean
        return -y.reshape(())

    model = Model(prior_model=prior_model, log_likelihood=log_likelihood)
    return model
