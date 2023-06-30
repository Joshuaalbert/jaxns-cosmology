#!/usr/bin/env python
import os

import bilby
import matplotlib.pyplot as plt

from jaxns_cosmology import install_jaxns_sampler
# installs the Jaxns sampler
from jaxns_cosmology.likelihoods import GaussianShells, EggBox, Himmelblau, Rastrigin, Rosenbrock

install_jaxns_sampler()

# To run GPU trained model with CPU device
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

try:
    from tensorflow.keras.models import load_model
except ImportError:
    pass

if __name__ == '__main__':
    # Output folders

    # Functions are:
    # 6D-CMB, 12D-MSSM7, nD-Rosenbrock, nD-Rastrigin, 2D-Himmelblau, nD-EggBox, nD-GaussianShells

    # Input is the number of dimensions for nD functions
    for likelihood in (Rastrigin(dimensionality=10), Rosenbrock(), GaussianShells(), EggBox(), Himmelblau()):
        outdir = likelihood.__class__.__name__
        priors = likelihood.priors

        # And run sampler
        # result = bilby.run_sampler(
        #    likelihood=likelihood, priors=priors, sampler='dynamic_dynesty', npoints=1000,
        #    outdir=outdir, label=label, bound='multi', sample='unif')

        result = bilby.run_sampler(
            likelihood=likelihood,
            priors=priors,
            sampler='jaxns',
            outdir=outdir,
            label=likelihood.__class__.__name__
        )

        # print(result.posterior)

        result.plot_corner()

        plt.show()
