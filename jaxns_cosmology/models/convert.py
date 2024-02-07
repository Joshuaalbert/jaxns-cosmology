import jax
from jaxns import Model


def convert_model(model: Model):
    """
    Convert a model to a function that computes the likelihood from unit cube.

    Args:
        model: A jaxns model

    Returns:
        A function that takes a flat array of `U_ndims` i.i.d. samples of U[0,1] and returns the log-likelihood
    """
    forward = jax.jit(lambda U: model.forward(U, allow_nan=True))
    return forward
