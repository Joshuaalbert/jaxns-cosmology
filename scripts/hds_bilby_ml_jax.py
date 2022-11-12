#!/usr/bin/env python
import os

import bilby
import matplotlib.pyplot as plt

from jaxns_cosmology import install_jaxns_sampler
# installs the Jaxns sampler
from jaxns_cosmology.likelihoods import GaussianShells

install_jaxns_sampler()

# To run GPU trained model with CPU device
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

try:
    from tensorflow.keras.models import load_model
except ImportError:
    pass

if __name__ == '__main__':
    # Output folders
    label = 'rosenbrock'
    outdir = 'outdir'

    # Functions are:
    # 6D-CMB, 12D-MSSM7, nD-Rosenbrock, nD-Rastrigin, 2D-Himmelblau, nD-EggBox, nD-GaussianShells

    # Input is the number of dimensions for nD functions
    dim = 2
    likelihood = GaussianShells(dim)
    priors = likelihood.priors

    # And run sampler
    # result = bilby.run_sampler(
    #    likelihood=likelihood, priors=priors, sampler='dynamic_dynesty', npoints=1000,
    #    outdir=outdir, label=label, bound='multi', sample='unif')

    result = bilby.run_sampler(
        likelihood=likelihood, priors=priors,
        sampler='jaxns', outdir=outdir, label=label)

    print(result.posterior)

    result.plot_corner()

    plt.show()
