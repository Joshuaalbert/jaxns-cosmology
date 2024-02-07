import jax
import jax.numpy as jnp

from jaxns_cosmology.models.CMB import build_CMB_model
from jaxns_cosmology.models.MSSM7 import build_MSSM7_model
from jaxns_cosmology.models.convert import convert_model
from jaxns_cosmology.models.eggbox import build_eggbox_model
from jaxns_cosmology.models.rosenbrock import build_rosenbrock_model
from jaxns_cosmology.models.spikeslab import build_spikeslab_model


def test_build_eggbox_model():
    forward = convert_model(build_eggbox_model(ndim=10))
    U_test = jax.random.uniform(jax.random.PRNGKey(0), shape=(10,), dtype=jnp.float32)
    output = forward(U_test)
    assert output.shape == ()
    print(output)
    assert ~ jnp.isnan(output)


def test_build_rosenbrock_model():
    forward = convert_model(build_rosenbrock_model(ndim=10))
    U_test = jax.random.uniform(jax.random.PRNGKey(0), shape=(10,), dtype=jnp.float32)
    output = forward(U_test)
    assert output.shape == ()
    print(output)
    assert ~ jnp.isnan(output)


def test_build_spike_slab_model():
    forward = convert_model(build_spikeslab_model(ndim=10))
    U_test = jax.random.uniform(jax.random.PRNGKey(0), shape=(10,), dtype=jnp.float32)
    output = forward(U_test)
    assert output.shape == ()
    print(output)
    assert ~ jnp.isnan(output)


def test_build_CMB_model():
    forward = convert_model(build_CMB_model())
    U_test = jax.random.uniform(jax.random.PRNGKey(0), shape=(6,), dtype=jnp.float32)
    output = forward(U_test)
    assert output.shape == ()
    print(output)
    assert ~ jnp.isnan(output)


def test_build_MSSM7_model():
    forward = convert_model(build_MSSM7_model())
    U_test = jax.random.uniform(jax.random.PRNGKey(0), shape=(12,), dtype=jnp.float32)
    output = forward(U_test)
    assert output.shape == ()
    print(output)
    assert ~ jnp.isnan(output)
