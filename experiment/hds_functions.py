#!/usr/bin/env python
"""
All log-likelihoods for bilby runs
"""
import os
from sys import exit
import argparse
import math
import scipy.stats as stats
import numpy as np
import time
import matplotlib.pyplot as plt
import json

import bilby
import jax.numpy as jnp
import jax
from jax.scipy.linalg import det, solve

import tensorflow as tf

# To run GPU trained model with CPU device
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

try:
    from tensorflow.keras.models import load_model
except ImportError:
    pass


def get_arguments() -> argparse.Namespace:
    """
    Set up an ArgumentParser to get the command line arguments.

    Returns:
        A Namespace object containing all the command line arguments
        for the script.
    """

    # Set up parser
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument('--dim',
                        default=2,
                        type=int,
                        metavar='N',
                        help='Number of the dimensions of the likelihood'
                        'Default: 2')
    parser.add_argument('--nlive',
                        default=500,
                        type=int,
                        metavar='N',
                        help='Number of live points'
                        'Default: 500')
    parser.add_argument('--steps',
                        default=100,
                        type=int,
                        metavar='N',
                        help='Number of steps for ultranest'
                        'Default: 100')
    parser.add_argument('--dlogz',
                        default=0.1,
                        type=float,
                        metavar='N',
                        help='Evidence tolerance'
                        'Default: 0.5')
    parser.add_argument('--walks',
                        default=1000,
                        type=int,
                        metavar='N',
                        help='Number of walks'
                        'Default: 1000')
    parser.add_argument('--model',
                        default='Rosenbrock',
                        type=str,
                        metavar='PATH',
                        help='Model Name '
                             'Default: Rosenbrock.')
    parser.add_argument('--sampler',
                        default='dynesty',
                        type=str,
                        metavar='PATH',
                        help='Sampler Name '
                             'Default: dynesty.')    
    parser.add_argument('--mode',
                        default='slice',
                        type=str,
                        metavar='PATH',
                        help='Sampler Name '
                             'Default: dynesty.')
    parser.add_argument('--outdir',
                        default='results2D',
                        type=str,
                        metavar='PATH',
                        help='Sampler Name '
                             'Default: results2D')
    parser.add_argument('--bound',
                        default='multi',
                        type=str,
                        metavar='PATH',
                        help='Sampler Name '
                             'Default: multi ellipsoidal')
    parser.add_argument('--sample',
                        default='rslice',
                        type=str,
                        metavar='PATH',
                        help='Sampler Name '
                             'Default: rslice')
    
    # Parse and return the arguments (as a Namespace object)
    arguments = parser.parse_args()
    return arguments

class MLFunction():
    """ Base class for functions that use a machine learning algorithm to
    provide function values """
    def __init__(self, *args, **kwargs):
        # Check if TF is installed
        try:
            load_model
        except NameError:
            raise ImportError(
                "The `tensorflow` package is not installed. This is needed in "
                "order to run `MLFunction`s. See the wiki on our GitHub "
                "project for installation instructions.")
        # Define object properties
        self.packageloc = self._get_package_location()
        self.model = None
        # Check definitions in parent class
        if not hasattr(self, 'modelname'):
            self.modelname = []
            raise Exception("MLFunction should define modelname.")

        is_standardised = hasattr(self, 'x_mean') and hasattr(self, 'x_stdev')
        if not is_standardised:
            self.x_mean, self.x_stdev = None, None
            raise Exception(
                "MLFunctions should either define x_mean and x_stdev or "
                "x_min and x_max.")

        is_standardised = hasattr(self, 'y_mean') and hasattr(self, 'y_stdev')
        if not is_standardised:
            self.y_mean, self.y_stdev = None, None
            raise Exception(
                "MLFunctions should either define y_mean and y_stdev or "
                "y_min and y_max.")

        super(MLFunction, self).__init__(*args, **kwargs)

    def _get_package_location(self):
        """ Get location in which the package was installed """
        this_dir, _ = os.path.split(__file__)
        return this_dir

    def _load_model(self):
        """ Load ML model from package """
        model_path = "ml_functions/{}/{}".format(self.modelname,
                                                    "model.hdf5")
        self.model = load_model(model_path)

    def _normalise(self, x, mu, sigma):
        return (x - mu) / sigma

    def _unnormalise(self, x, mu, sigma):
        return x * sigma + mu

    def _evaluate(self, x):
        # Load model is not already done
        if self.model is None:
            self._load_model()
        # Correct shape of x is needed
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        # Query to model
        x = self._normalise(x, self.x_mean, self.x_stdev)
        y = self.model.predict(x)
        y = self._unnormalise(y, self.y_mean, self.y_stdev)
        return y

class CMB_mod(MLFunction, bilby.Likelihood):
    """
    N-dimensional Rosenbrock as defined in
    https://en.wikipedia.org/wiki/Rosenbrock_function

    Args:
        dimensionality: Number of input dimensions the function should take.

    """

    def __init__(self):
        self.modelname = 'lcdm'
        self.dim = 6
        self.x_mean = np.array([
            0.95985349, 1.04158987, 0.02235954, 0.11994063, 0.05296935,
            3.06873361], np.float64)
        self.x_stdev = np.array([
            0.00836984, 0.00168727, 0.00043045, 0.00470686, 0.00899083,
            0.02362278], np.float64)

        self.y_mean = 381.7929565376543
        self.y_stdev = 1133.7707883293974


        # Limit sampling between these hard borders, in order to ALWAYS remain
        # within the training box.
        ranges = []

        x_min = [0.955, 1.037, 0.02, 0.1, 0.04, 3.09
        ]
        x_max = [0.99, 1.042, 0.02255, 0.1225, 0.097, 3.16
        ]

        for i in range(len(x_min)):
            ranges.append([x_min[i], x_max[i]])
        self.ranges = ranges

        """ 
        parameters = {'omega_b':0, 'omega_cdm':0, 'theta_s':0, 'ln[A_s]':0, 'n_s':0, 'tau_r':0}
        labels = np.array(['$\\omega_b$', '$\\omega_{cdm}$', '$\\theta_s$', '$ln[A_s]$', '$n_s$', '$\\tau_r$'])
        self.names = np.array(['omega_b', 'omega_cdm', 'theta_s', 'ln[A_s]', 'n_s', 'tau_r'])
        """
        
        parameters = {'n_s':0, 'theta_s':0, 'omega_b':0, 'omega_cdm':0, 'tau_r':0, 'ln[A_s]':0}
        labels = np.array(['$n_s$', '$\\theta_s$', '$\\omega_b$', '$\\omega_{cdm}$', '$\\tau_r$', '$ln[A_s]$'])
        self.names = np.array(['n_s', 'theta_s', 'omega_b', 'omega_cdm', 'tau_r', 'ln[A_s]'])

        #Uniform priors are assumed
        priors = dict()
        i = 0
        for key in parameters:
          priors[key] = bilby.core.prior.Uniform(ranges[i][0], ranges[i][1], labels[i])
          i += 1

        self.priors = priors
        super(CMB, self).__init__(parameters=parameters)

    def log_likelihood(self):
        if self.model is None:
            self._load_model()
        x = np.array([self.parameters[self.names[i]] for i in range(self.dim)])
        cut = False
        x[0] -= 0.002
        x[1] -= 0.0005
        x[3] -= 0.001
        x[4] += 0.02
        if x[0] < 0.955 or x[1] >  1.0412 or x[2] > 0.0225:
          cut = True
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        x = self._normalise(x, self.x_mean, self.x_stdev)
        y = self.model.predict(x)
        y = self._unnormalise(y, self.y_mean, self.y_stdev)
        if cut:
          return -(y[0][0] - 0.35*y[0][0])
        return -y[0][0]

    
class CMB(MLFunction, bilby.Likelihood):
    """
    N-dimensional Rosenbrock as defined in
    https://en.wikipedia.org/wiki/Rosenbrock_function

    Args:
        dimensionality: Number of input dimensions the function should take.

    """

    def __init__(self):
        self.modelname = 'lcdm'
        self.dim = 6
        self.x_mean = np.array([

            0.95985349, 1.04158987, 0.02235954, 0.11994063, 0.05296935,
            3.06873361], np.float64)
        self.x_stdev = np.array([
            0.00836984, 0.00168727, 0.00043045, 0.00470686, 0.00899083,
            0.02362278], np.float64)

        self.y_mean = 381.7929565376543
        self.y_stdev = 1133.7707883293974


        # Limit sampling between these hard borders, in order to ALWAYS remain
        # within the training box.
        ranges = []

        #x_min = [0.92, 1.037, 0.02, 0.1, 0.0100002, 2.98
        #]
        #x_max = [0.999999, 1.05, 0.024, 0.14, 0.097, 3.16
        #]

        x_min = [0.954, 1.037, 0.02, 0.1, 0.05, 3.05
        ]
        x_max = [0.966, 1.042, 0.0226, 0.14, 0.097, 3.16
        ]
        
        for i in range(len(x_min)):
            ranges.append([x_min[i], x_max[i]])
        self.ranges = ranges

        """ 
        parameters = {'omega_b':0, 'omega_cdm':0, 'theta_s':0, 'ln[A_s]':0, 'n_s':0, 'tau_r':0}
        labels = np.array(['$\\omega_b$', '$\\omega_{cdm}$', '$\\theta_s$', '$ln[A_s]$', '$n_s$', '$\\tau_r$'])
        self.names = np.array(['omega_b', 'omega_cdm', 'theta_s', 'ln[A_s]', 'n_s', 'tau_r'])
        """
        
        parameters = {'n_s':0, 'theta_s':0, 'omega_b':0, 'omega_cdm':0, 'tau_r':0, 'ln[A_s]':0}
        labels = np.array(['$n_s$', '$\\theta_s$', '$\\omega_b$', '$\\omega_{cdm}$', '$\\tau_r$', '$ln[A_s]$'])
        self.names = np.array(['n_s', 'theta_s', 'omega_b', 'omega_cdm', 'tau_r', 'ln[A_s]'])

        #Uniform priors are assumed
        priors = dict()
        i = 0
        for key in parameters:
          priors[key] = bilby.core.prior.Uniform(ranges[i][0], ranges[i][1], labels[i])
          i += 1

        self.priors = priors
        super(CMB, self).__init__(parameters=parameters)

    def log_likelihood(self):
        if self.model is None:
            self._load_model()
        x = np.array([self.parameters[self.names[i]] for i in range(self.dim)])
        cut = False
        #x[0] -= 0.002
        #x[1] -= 0.0005
        #x[3] -= 0.001
        x[4] += 0.01
        x[5] += 0.02
        if x[2] > 0.0224:
          x[2] = 0.0446 - 1.*x[2]  
          #cut = True
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        x = self._normalise(x, self.x_mean, self.x_stdev)
        y = self.model.predict(x)
        y = self._unnormalise(y, self.y_mean, self.y_stdev)
        #if cut:
        #  return -(y[0][0] - 0.28*y[0][0])
        return -y[0][0]

class CMB_jax(MLFunction, bilby.Likelihood):
    """
    N-dimensional Rosenbrock as defined in
    https://en.wikipedia.org/wiki/Rosenbrock_function

    Args:
        dimensionality: Number of input dimensions the function should take.

    """

    def __init__(self):
        self.modelname = 'lcdm'
        self.dim = 6
        self.x_mean = jnp.array([
            0.95985349, 1.04158987, 0.02235954, 0.11994063, 0.05296935,
            3.06873361], jnp.float64)
        self.x_stdev = jnp.array([
            0.00836984, 0.00168727, 0.00043045, 0.00470686, 0.00899083,
            0.02362278], jnp.float64)

        self.y_mean = 381.7929565376543
        self.y_stdev = 1133.7707883293974


        # Limit sampling between these hard borders, in order to ALWAYS remain
        # within the training box.
        ranges = []

        #x_min = [0.92, 1.037, 0.02, 0.1, 0.0100002, 2.98
        #]
        #x_max = [0.999999, 1.05, 0.024, 0.14, 0.097, 3.16
        #]

        x_min = [0.954, 1.037, 0.02, 0.1, 0.05, 3.05
        ]
        x_max = [0.966, 1.042, 0.0226, 0.14, 0.097, 3.16
        ]
        
        for i in range(len(x_min)):
            ranges.append([x_min[i], x_max[i]])
        self.ranges = ranges

        """ 
        parameters = {'omega_b':0, 'omega_cdm':0, 'theta_s':0, 'ln[A_s]':0, 'n_s':0, 'tau_r':0}
        labels = np.array(['$\\omega_b$', '$\\omega_{cdm}$', '$\\theta_s$', '$ln[A_s]$', '$n_s$', '$\\tau_r$'])
        self.names = np.array(['omega_b', 'omega_cdm', 'theta_s', 'ln[A_s]', 'n_s', 'tau_r'])
        """
        
        parameters = {'n_s':0, 'theta_s':0, 'omega_b':0, 'omega_cdm':0, 'tau_r':0, 'ln[A_s]':0}
        #labels = np.array(['$n_s$', '$\\theta_s$', '$\\omega_b$', '$\\omega_{cdm}$', '$\\tau_r$', '$ln[A_s]$'])
        labels = np.array(['n_s', 'theta_s', 'omega_b', 'omega_cdm', 'tau_r', 'A_s'])
        self.names = np.array(['n_s', 'theta_s', 'omega_b', 'omega_cdm', 'tau_r', 'ln[A_s]'])

        #print(labels)
        #Uniform priors are assumed
        priors = dict()
        i = 0
        for key in parameters:
          priors[key] = bilby.core.prior.Uniform(ranges[i][0], ranges[i][1], labels[i])
          #priors[key] = bilby.core.prior.Uniform(ranges[i][0], ranges[i][1])     
          i += 1
        self.priors = priors
        super(CMB_jax, self).__init__(parameters=parameters)

    def condition(self, x):
     return x[2] > 0.0224

    def true_fun(self, x):
     return x.at[2].set(0.0446 - 1.0 * x[2])

    def false_fun(self, x):
     return x
 
    def log_likelihood(self):
        if self.model is None:
            self._load_model()
        x = jnp.array([self.parameters[self.names[i]] for i in range(self.dim)])
        x = x.at[4].add(0.01)
        x = x.at[5].add(0.02)
        #if x[2] > 0.0224:
        #  x = x.at[2].set(0.0446 - 1.0 * x[2])
        x = jax.lax.cond(self.condition(x), self.true_fun, self.false_fun, operand=x)
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        x = self._normalise(x, self.x_mean, self.x_stdev)
        x = tf.convert_to_tensor(x)
        y = self.model.predict(x)
        y = self._unnormalise(y, self.y_mean, self.y_stdev)
        return -y[0][0]
    
class MSSM7(MLFunction, bilby.Likelihood):
    """
    N-dimensional Rosenbrock as defined in
    https://en.wikipedia.org/wiki/Rosenbrock_function

    Args:
        dimensionality: Number of input dimensions the function should take.

    """

    def __init__(self):
        self.modelname = 'mssm7'
        self.dim = 12
        self.x_mean = np.array([
            -1.65550622e+02, 6.53242357e+07, -6.04267288e+06, 1.15227686e+07,
            -8.96546390e+02, 1.20880748e+03, 3.65456629e+01, 1.73423279e+02,
            1.18539912e-01, 4.00306869e-01, 4.31081695e+01, 5.80441328e+01
        ], np.float32)
        self.x_stdev = np.array([
            3.13242671e+03, 2.64037878e+07, 4.32735828e+06, 2.22328069e+07,
            2.33891832e+03, 6.20060930e+03, 1.40566829e+01, 3.83628710e-01,
            2.51362291e-04, 4.59609868e-02, 3.09244370e+00, 3.24780776e+00
        ], np.float32)
        self.y_mean = -262.5887645450105
        self.y_stdev = 7.461633956842537

        # Limit sampling between these hard borders, in order to ALWAYS remain
        # within the training box.
        ranges = []

        x_min = [
            -7.16775760e+03, 4.27547804e+05, -9.98192815e+07, -6.81824964e+07,
            -9.99995488e+03, -9.99999903e+03, 3.00597043e+00, 1.71060011e+02,
            1.16700013e-01, 2.00000156e-01, 1.90001455e+01, 3.10001673e+01
        ]
        x_max = [
            7.18253463e+03, 9.99999857e+07, 4.56142832e+05, 9.99999734e+07,
            9.99987623e+03, 9.99999881e+03, 6.99999394e+01, 1.75619963e+02,
            1.20299997e-01, 7.99999435e-01, 6.69997800e+01, 8.49983345e+01
        ]

        for i in range(len(x_min)):
            ranges.append([x_min[i], x_max[i]])
        self.ranges = ranges

        parameters = {'M2':0, 'mf2':0, 'mHu2':0, 'mHd2':0, 'Au':0, 'Ad':0, 'tanb':0, 'mt':0, 'alphas':0, 'rho0':0, 'sigmas':0, 'sigmal':0}
        labels = np.array(['$M_2$', '$m_f^2$', '$m_{H_u}^2$', '$m_{H_d}^2$', '$A_u$', '$A_d$', 'tan \\beta', '$m_t$', '$\\alpha_s$', '$\\rho_0$', '$\\sigma_s$', '$\\sigma_l$'])
        self.names = np.array(['M2', 'mf2', 'mHu2', 'mHd2', 'Au', 'Ad', 'tanb', 'mt', 'alphas', 'rho0', 'sigmas', 'sigmal'])

        #Uniform priors are assumed
        priors = dict()
        i = 0
        for key in parameters:
          priors[key] = bilby.core.prior.Uniform(ranges[i][0], ranges[i][1], labels[i])
          i += 1

        self.priors = priors

        super(MSSM7, self).__init__(parameters=parameters)

    def log_likelihood(self):
        if self.model is None:
            self._load_model()
        x = np.array([self.parameters[self.names[i]] for i in range(self.dim)])
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        x = self._normalise(x, self.x_mean, self.x_stdev)
        y = self.model.predict(x)
        y = self._unnormalise(y, self.y_mean, self.y_stdev)
        return y[0][0]

class Rosenbrock(bilby.Likelihood):
    """
    N-dimensional Rosenbrock as defined in
    https://en.wikipedia.org/wiki/Rosenbrock_function
 
    We take a = 1, b = 100

    Args:
        dimensionality: Number of input dimensions the function should take.

    """

    def __init__(self, dimensionality=2):
        if dimensionality < 2:
            raise Exception("""Dimensionality of Rosenbrock function has to
                            be >=2.""")
        self.dim = dimensionality

        x_min = -5
        x_max = 5

        ranges = []
        for i in range(self.dim):
            ranges.append([x_min, x_max])
        self.ranges = ranges

        parameters = {"x{0}".format(i): 0 for i in range(self.dim)}

        #Uniform priors are assumed
        priors = bilby.core.prior.PriorDict()
        priors.update({"x{0}".format(i): bilby.core.prior.Uniform(ranges[i][0], ranges[i][1], "x{0}".format(i)) for i in range(dim)})
        
        self.priors = priors

        super(Rosenbrock, self).__init__(parameters=parameters)

    def log_likelihood(self):
        x = np.array([self.parameters["x{0}".format(i)] for i in range(self.dim)])
        y = 0
        for i in range(self.dim - 1):
            y += (100 * np.power(x[i + 1] - np.power(x[i], 2), 2) +
                  np.power(1 - x[i], 2))
        return -y

class Rosenbrock_jax(bilby.Likelihood):
    """
    N-dimensional Rosenbrock as defined in
    https://en.wikipedia.org/wiki/Rosenbrock_function
 
    We take a = 1, b = 100

    Args:
        dimensionality: Number of input dimensions the function should take.

    """

    def __init__(self, dimensionality=2):
        if dimensionality < 2:
            raise Exception("""Dimensionality of Rosenbrock function has to
                            be >=2.""")
        self.dim = dimensionality

        x_min = -5
        x_max = 5

        ranges = []
        for i in range(self.dim):
            ranges.append([x_min, x_max])
        self.ranges = ranges

        parameters = {"x{0}".format(i): 0 for i in range(self.dim)}

        #Uniform priors are assumed
        priors = bilby.core.prior.PriorDict()
        priors.update({"x{0}".format(i): bilby.core.prior.Uniform(ranges[i][0], ranges[i][1], "x{0}".format(i)) for i in range(dim)})
        
        self.priors = priors

        super(Rosenbrock_jax, self).__init__(parameters=parameters)

    def log_likelihood(self):
        x = jnp.array([self.parameters["x{0}".format(i)] for i in range(self.dim)])
        y = 0
        for i in range(self.dim - 1):
            y += (100 * jnp.power(x[i + 1] - jnp.power(x[i], 2), 2) +
                  jnp.power(1 - x[i], 2))
        return -y

class Rastrigin(bilby.Likelihood):
    """
    N-dimensional Rastrigin function as defined in
    https://en.wikipedia.org/wiki/Rastrigin_function

    We take A = 100

    Args:
        dimensionality: Number of input dimensions the function should take.
    """

    def __init__(self, dimensionality=2):
        if dimensionality < 2:
            raise Exception("""Dimensionality of Rastrigin function has to
                            be >=2.""")
        self.a = 10
        self.dim = dimensionality

        x_min = -5.12
        x_max = 5.12

        ranges = []
        for i in range(self.dim):
            ranges.append([x_min, x_max])
        self.ranges = ranges

        parameters = {"x{0}".format(i): 0 for i in range(self.dim)}

        #Uniform priors are assumed
        priors = bilby.core.prior.PriorDict()
        priors.update({"x{0}".format(i): bilby.core.prior.Uniform(ranges[i][0], ranges[i][1], "x{0}".format(i)) for i in range(dim)})

        self.priors = priors

        super(Rastrigin, self).__init__(parameters=parameters)
                
    def log_likelihood(self):
        x = np.array([self.parameters["x{0}".format(i)] for i in range(self.dim)])             
        y = self.a * self.dim
        for i in range(self.dim):
            y += np.power(x[i], 2) - self.a * np.cos(2 * np.pi * x[i])

        return -y

class Rastrigin_jax(bilby.Likelihood):
    """
    N-dimensional Rastrigin function as defined in
    https://en.wikipedia.org/wiki/Rastrigin_function

    We take A = 100

    Args:
        dimensionality: Number of input dimensions the function should take.
    """

    def __init__(self, dimensionality=2):
        if dimensionality < 2:
            raise Exception("""Dimensionality of Rastrigin function has to
                            be >=2.""")
        self.a = 10
        self.dim = dimensionality

        x_min = -5.12
        x_max = 5.12

        ranges = []
        for i in range(self.dim):
            ranges.append([x_min, x_max])
        self.ranges = ranges

        parameters = {"x{0}".format(i): 0 for i in range(self.dim)}

        #Uniform priors are assumed
        priors = bilby.core.prior.PriorDict()
        priors.update({"x{0}".format(i): bilby.core.prior.Uniform(ranges[i][0], ranges[i][1], "x{0}".format(i)) for i in
 range(dim)})

        self.priors = priors

        super(Rastrigin_jax, self).__init__(parameters=parameters)
                
    def log_likelihood(self):
        x = jnp.array([self.parameters["x{0}".format(i)] for i in range(self.dim)])
        y = self.a * self.dim
        for i in range(self.dim):
            y += jnp.power(x[i], 2) - self.a * jnp.cos(2 * jnp.pi * x[i])

        return -y

class Himmelblau(bilby.Likelihood):
    """
    Himmelblau function as defined in
    https://en.wikipedia.org/wiki/Himmelblau%27s_function

    This is a 2-dimensional function with an application range bounded by -5
    and 5 for both input variables. 

    """

    def __init__(self, dimensionality=2):
        self.dim = 2

        x_min = -6
        x_max = 6

        ranges = []
        for i in range(self.dim):
            ranges.append([x_min, x_max])
        self.ranges = ranges

        parameters = {"x{0}".format(i): 0 for i in range(self.dim)}

        #Uniform priors are assumed
        priors = bilby.core.prior.PriorDict()
        priors.update({"x{0}".format(i): bilby.core.prior.Uniform(ranges[i][0], ranges[i][1], "x{0}".format(i)) for i in range(dim)})

        self.priors = priors

        super(Himmelblau, self).__init__(parameters=parameters)

    def log_likelihood(self):
        x = np.array([self.parameters["x{0}".format(i)] for i in range(self.dim)])
        y = np.power(np.power(x[0], 2) + x[1] - 11, 2) + np.power(x[0] + np.power(x[1], 2) - 7, 2)
        return -y
    
class Himmelblau_jax(bilby.Likelihood):
    """
    Himmelblau function as defined in
    https://en.wikipedia.org/wiki/Himmelblau%27s_function

    This is a 2-dimensional function with an application range bounded by -5
    and 5 for both input variables. 

    """

    def __init__(self, dimensionality=2):
        self.dim = 2

        x_min = -6
        x_max = 6

        ranges = []
        for i in range(self.dim):
            ranges.append([x_min, x_max])
        self.ranges = ranges

        parameters = {"x{0}".format(i): 0 for i in range(self.dim)}

        #Uniform priors are assumed
        priors = bilby.core.prior.PriorDict()
        priors.update({"x{0}".format(i): bilby.core.prior.Uniform(ranges[i][0], ranges[i][1], "x{0}".format(i)) for i in
 range(dim)})

        self.priors = priors

        super(Himmelblau_jax, self).__init__(parameters=parameters)

    def log_likelihood(self):
        x = jnp.array([self.parameters["x{0}".format(i)] for i in range(self.dim)])
        y = jnp.power(jnp.power(x[0], 2) + x[1] - 11, 2) + jnp.power(x[0] + jnp.power(x[1], 2) - 7, 2)
        return -y


class EggBox(bilby.Likelihood):
    """
    N-dimensional EggBox as defined in arXiv:0809.3437.
    Testfunction as defined in arXiv:0809.3437

    Args:
        dimensionality: Number of input dimensions the function should take.

    """

    def __init__(self, dimensionality=2):
        if dimensionality < 2:
            raise Exception("""Dimensionality of EggBox function has to
                            be >=2.""")
        self.dim = dimensionality
        self.tmax = 5.0 * np.pi

        x_min = 0.
        x_max = 10.*np.pi

        ranges = []
        for i in range(self.dim):
            ranges.append([x_min, x_max])
        self.ranges = ranges

        parameters = {"x{0}".format(i): 0 for i in range(self.dim)}

        #Uniform priors are assumed
        priors = bilby.core.prior.PriorDict()
        priors.update({"x{0}".format(i): bilby.core.prior.Uniform(ranges[i][0], ranges[i][1], "x{0}".format(i)) for i in range(dim)})

        self.priors = priors

        super(EggBox, self).__init__(parameters=parameters)

    def log_likelihood(self):
        x = np.array([self.parameters["x{0}".format(i)] for i in range(self.dim)])
        y = 1

        for i in range(self.dim):
            y *= math.cos(x[i]/2)
        y = math.pow(2. + y, 5)
        return y

class EggBox_jax(bilby.Likelihood):
    """
    N-dimensional EggBox as defined in arXiv:0809.3437.
    Testfunction as defined in arXiv:0809.3437

    Args:
        dimensionality: Number of input dimensions the function should take.

    """

    def __init__(self, dimensionality=2):
        if dimensionality < 2:
            raise Exception("""Dimensionality of EggBox function has to
                            be >=2.""")
        self.dim = dimensionality
        #self.tmax = 5.0 * jnp.pi

        x_min = 0.
        x_max = 10.*jnp.pi

        ranges = []
        for i in range(self.dim):
            ranges.append([x_min, x_max])
        self.ranges = ranges

        parameters = {"x{0}".format(i): 0 for i in range(self.dim)}

        #Uniform priors are assumed
        priors = bilby.core.prior.PriorDict()
        priors.update({"x{0}".format(i): bilby.core.prior.Uniform(ranges[i][0], ranges[i][1], "x{0}".format(i)) for i in range(dim)})

        self.priors = priors

        super(EggBox_jax, self).__init__(parameters=parameters)

    def log_likelihood(self):
        x = jnp.array([self.parameters["x{0}".format(i)] for i in range(self.dim)])
        y = 1

        for i in range(self.dim):
         y *= jnp.cos(0.5*x[i])
        y = jnp.power(2. + y, 5)
        #y = 5. * (2. + y)
        return y


class GaussianShells(bilby.Likelihood):

    """
    N-dimensional GaussianShells as defined in arXiv:0809.3437.
    Both number of dimensions and number of rings are arbitrary but we assume that have 
    an equal radius and widths 

    Args:
        dimensionality: Number of input dimensions the function should take.
        modes: number of rings
        r: radius of rings
        w: width of rings
        c: is a n-rings x n-dims list with centers coordinates
    """

    def __init__(self, dimensionality=2, modes=2, r=2, w=0.1, c1 = -3.5, c2 = 3.5):
         if dimensionality < 2:
            raise Exception("""Dimensionality of GaussianShells function has to
                            be >=2.""")

         self.dim = dimensionality
         self.modes = modes
         self.r = r
         self.w = w

         self.c1_arr = np.zeros(dimensionality)
         self.c2_arr = np.zeros(dimensionality)
         self.c1_arr[0] = c1
         self.c2_arr[0] = c2

         self.const = math.log(1. / math.sqrt(2. * math.pi * w**2))  # normalization constant

         x_min = -6
         x_max = 6

         ranges = []
         for i in range(self.dim):
            ranges.append([x_min, x_max])
         self.range = ranges

         parameters = {"x{0}".format(i): 0 for i in range(self.dim)}

         #Uniform priors are assumed
         priors = bilby.core.prior.PriorDict()
         priors.update({"x{0}".format(i): bilby.core.prior.Uniform(ranges[i][0], ranges[i][1], "x{0}".format(i)) for i in range(dim)})

         self.priors = priors

         super(GaussianShells, self).__init__(parameters=parameters)


    # log-likelihood of a single shell
    def logcirc(self, theta, c):
         d = np.sqrt(np.sum((theta - c)**2, axis=-1))  # |theta - c|
         return self.const - (d - self.r)**2 / (2. * self.w**2)

    def log_likelihood(self):
     x = np.array([self.parameters["x{0}".format(i)] for i in range(self.dim)])
     return np.logaddexp(self.logcirc(x, self.c1_arr), self.logcirc(x, self.c2_arr))

    """
    def log_likelihood(self):
     x = np.array([self.parameters["x{0}".format(i)] for i in range(self.dim)])
     y = -240.
     for i in range(self.modes):
      y = np.logaddexp(y, self.logcirc(x, self.c[i]))
     return y
    """

class GaussianShells_jax(bilby.Likelihood):

    """
    N-dimensional GaussianShells as defined in arXiv:0809.3437.
    Both number of dimensions and number of rings are arbitrary but we assume that have 
    an equal radius and widths 

    Args:
        dimensionality: Number of input dimensions the function should take.
        modes: number of rings
        r: radius of rings
        w: width of rings
        c: is a n-rings x n-dims list with centers coordinates
    """

    def __init__(self, dimensionality=2, modes=2, r=2, w=0.1, c1 = -3.5, c2 = 3.5): 

         if dimensionality < 2:
            raise Exception("""Dimensionality of GaussianShells function has to
                            be >=2.""")

         self.dim = dimensionality
         self.modes = modes
         self.r = r
         self.w = w

         self.c1_arr = jnp.zeros(dimensionality)
         self.c2_arr = jnp.zeros(dimensionality)
         self.c1_arr = self.c1_arr.at[0].set(c1)
         self.c2_arr = self.c2_arr.at[0].set(c2)

         self.const = jnp.log(1. / jnp.sqrt(2. * jnp.pi * w**2))  # normalization constant

         x_min = -6
         x_max = 6

         ranges = []
         for i in range(self.dim):
            ranges.append([x_min, x_max])
         self.ranges = ranges

         parameters = {"x{0}".format(i): 0 for i in range(self.dim)}

         #Uniform priors are assumed
         priors = bilby.core.prior.PriorDict()
         priors.update({"x{0}".format(i): bilby.core.prior.Uniform(ranges[i][0], ranges[i][1], "x{0}".format(i)) for i in range(dim)})

         self.priors = priors

         super(GaussianShells_jax, self).__init__(parameters=parameters)


    # log-likelihood of a single shell
    def logcirc(self, theta, c):
         d = jnp.linalg.norm(theta - c) #np.sqrt(np.sum((theta - c)**2, axis=-1))  # |theta - c|
         return self.const - (d - self.r)**2 / (2. * self.w**2)

    def log_likelihood(self):
     x = jnp.array([self.parameters["x{0}".format(i)] for i in range(self.dim)])
     return jnp.logaddexp(self.logcirc(x, self.c1_arr), self.logcirc(x, self.c2_arr))

    """
    def log_likelihood(self):
     x = jnp.array([self.parameters["x{0}".format(i)] for i in range(self.dim)])
     #x = jnp.array([-1.314188,4.6734276])
     #jax.debug.print("x: {}", x)
     
     y = -1.e4
     for i in range(self.modes):
      y = jnp.logaddexp(y, self.logcirc(x, self.c[i]))
     #jax.debug.print("y: {}", y)
     #y = jnp.logaddexp(self.logcirc(x, self.c[0]), self.logcirc(x, self.c[1]))
     return y
    """

class SpikeSlab(bilby.Likelihood):

    def __init__(self, dimensionality=2):
        if dimensionality < 2:
            raise Exception("""Dimensionality of SpikeSlab function has to
                            be >=2.""")

        self.dim = dimensionality

        x_min = -4
        x_max = 8.

        ranges = []
        for i in range(self.dim):
            ranges.append([x_min, x_max])
        self.ranges = ranges

        parameters = {"x{0}".format(i): 0 for i in range(self.dim)}

        #Uniform priors are assumed
        priors = bilby.core.prior.PriorDict()
        priors.update({"x{0}".format(i): bilby.core.prior.Uniform(ranges[i][0], ranges[i][1], "x{0}".format(i)) for i in range(dim)})

        self.priors = priors

        super(SpikeSlab, self).__init__(parameters=parameters)

    def log_likelihood(self):
        x = np.array([self.parameters["x{0}".format(i)] for i in range(self.dim)])
        mean_1 = [6,6]
        mean_2 = [2.5,2.5]
        for i in range(self.dim-2):
            mean_1.append(0)
            mean_2.append(0)
        cov_1 = 0.08*np.identity(self.dim)
        cov_2 = 0.8*np.identity(self.dim)
        gauss_1 = stats.multivariate_normal.pdf(x,mean = mean_1,cov = cov_1)
        gauss_2 = stats.multivariate_normal.pdf(x,mean = mean_2,cov = cov_2)
        y = np.log(gauss_1 + gauss_2)
        return y

class SpikeSlab_jax(bilby.Likelihood):

    def __init__(self, dimensionality=2):
        if dimensionality < 2:
            raise Exception("""Dimensionality of SpikeSlab function has to
                            be >=2.""")

        self.dim = dimensionality

        x_min = -4
        x_max = 8.

        ranges = []
        for i in range(self.dim):
            ranges.append([x_min, x_max])
        self.ranges = ranges

        parameters = {"x{0}".format(i): 0 for i in range(self.dim)}

        #Uniform priors are assumed
        priors = bilby.core.prior.PriorDict()
        priors.update({"x{0}".format(i): bilby.core.prior.Uniform(ranges[i][0], ranges[i][1], "x{0}".format(i)) for i in range(dim)})

        self.priors = priors

        super(SpikeSlab_jax, self).__init__(parameters=parameters)

    def multivariate_normal_pdf(self, x, mean, cov):
        """Compute the multivariate normal pdf."""
        dim = x.shape[0]
        prefactor = 1. / (jnp.sqrt((2 * jnp.pi) ** dim * det(cov)))
        deviation = x - mean
        exponent = -0.5 * jnp.dot(deviation, solve(cov, deviation))
        return prefactor * jnp.exp(exponent)
        
    def log_likelihood(self):
        x = jnp.array([self.parameters["x{0}".format(i)] for i in range(self.dim)])
        mean_1 = [6, 6] + [0] * (self.dim - 2)
        mean_2 = [2.5, 2.5] + [0] * (self.dim - 2)
        cov_1 = 0.08 * jnp.identity(self.dim)
        cov_2 = 0.8 * jnp.identity(self.dim)
        gauss_1 = self.multivariate_normal_pdf(x, jnp.array(mean_1), cov_1)
        gauss_2 = self.multivariate_normal_pdf(x, jnp.array(mean_2), cov_2)
        y = jnp.log(gauss_1 + gauss_2)
        return y

    
# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == '__main__':


 print('')
 print('Sampling a model')
 print('')

 # Start the stopwatch
 script_start = time.time()

 # Read in command line arguments
 args = get_arguments()

 #Functions are:
 #6D-CMB, 12D-MSSM7, nD-Rosenbrock, nD-Rastrigin, 2D-Himmelblau, nD-EggBox, nD-GaussianShells

 #Input is the number of dimensions for nD functions 
 dim = args.dim
 nlive = args.nlive
 dlogz = args.dlogz
 nsteps = args.steps
 walks = args.walks
 model = args.model
 sampler = args.sampler
 outdir = args.outdir
 mode = args.mode
 bound = args.bound
 sample = args.sample

 #Check whether outdir already exits
 if not os.path.isdir(outdir):
    os.makedirs(outdir)

 dynesty = dict(nlive=nlive, dlogz=dlogz, bound=bound, sample=sample, naccept=50, walks=walks) 
 dynamic_dynesty = dict(nlive_init=nlive, dlogz_init=dlogz, live_batch=1000, bound=bound, sample=sample, walks=walks)
 pymultinest = dict(importance_nested_sampling=False, nlive=nlive, evidence_tolerance=dlogz, multimodal=True, max_modes=100, max_iter=0, soft_init=True)
 pypolychord = dict(nlive=nlive, num_repeats=2*dim, precision_criterion=dlogz, soft_init=True, do_clustering=True)
 nessai = dict(nlive=nlive, stopping=dlogz, vectorized=True, soft_init=True)
 #If the number of live points is specified the `ultranest.NestedSampler` will be used, otherwise the
 #`ultranest.ReactiveNestedSampler` will be used.
 if nlive == 0:
  ultranest = dict(max_num_improvement_loops=1, dlogz=dlogz) #num_bootstraps=1)ls
  if mode == 'slice':
     #step_sampler = 'RegionSliceSampler' + '(nsteps=100, adaptive_nsteps=adaptive_nsteps, max_nsteps=dim, log=open(outdir' + '/stepsampler.log' + ',' + 'w' + '))'
     step_sampler = 'ultranest.stepsampler.RegionSliceSampler(nsteps=' + str(nsteps) + ', max_nsteps=' + str(dim) + ', log=open("' + outdir + '/stepsampler.log", ' + '"w"' + '))'
  if mode == 'harm':
     step_sampler = 'ultranest.stepsampler.RegionBallSliceSampler(nsteps=100, max_nsteps=' + str(dim) + ', log=open("' + outdir + '/stepsampler.log", ' + '"w"' + '))'
  if mode == 'aharm':
     step_sampler = 'ultranest.stepsampler.AHARMSampler(nsteps=200, max_nsteps=' + str(dim) + ', log=open("' + outdir + '/stepsampler.log", ' + '"w"' + '))'

  ultranest = dict(max_num_improvement_loops=1, dlogz=dlogz, step_sampler = step_sampler)   
 else:
  ultranest = dict(nlive=nlive, dlogz=dlogz, max_num_improvement_loops=1)
 jaxns = dict(num_live_points=nlive, use_jaxns_defaults=False, max_samples=1.e6, uncert_improvement_patience=1, live_evidence_frac=dlogz)
 dnest4 = dict(num_steps=nlive, backend='csv', soft_init=True)
 nautilus = dict(n_live=nlive, n_networks=4, enlarge_per_dim=1.1, split_threshold=1.1, n_batch=100)
 
 #Samplers
 if sampler == 'dynesty':
   sampler_dict = dynesty
 elif sampler == 'dynamic_dynesty':
   sampler_dict = dynamic_dynesty
 elif sampler == 'pymultinest':
   sampler_dict = pymultinest
 elif sampler == 'pypolychord':
   sampler_dict = pypolychord
 elif sampler == 'nessai':
   sampler_dict = nessai
 elif sampler == 'ultranest':
   sampler_dict = ultranest
 elif sampler == 'dnest4':
   sampler_dict = dnest4
 elif sampler == 'nautilus':
   sampler_dict = nautilus  
 elif sampler == 'jaxns':
   sampler_dict = jaxns
 else:
    print('Non allowed sampler')
    exit()

 #Likelihoods
 if model == 'Rosenbrock':
  likelihood = Rosenbrock(dim)
  likelihood_jax = Rosenbrock_jax(dim)
 elif model == 'Rastrigin':
  likelihood = Rastrigin(dim)  
  likelihood_jax = Rastrigin_jax(dim)
 elif model == 'Himmelblau':
  likelihood = Himmelblau(dim)
  likelihood_jax = Himmelblau_jax(dim)
 elif model == 'EggBox':
  likelihood = EggBox(dim)
  likelihood_jax = EggBox_jax(dim)
 elif model == 'GaussianShells':
  likelihood = GaussianShells(dim)
  likelihood_jax = GaussianShells_jax(dim)
 elif model == 'SpikeSlab':
  likelihood = SpikeSlab(dim)
  likelihood_jax = SpikeSlab_jax(dim)   
 elif model == 'CMB':
  likelihood = CMB()
  likelihood_jax = CMB_jax()
 elif model == 'MSSM7':
  likelihood = MSSM7()
 else:
  print('Non allowed Likelihood model')
  exit()

 priors = likelihood.priors

 if sampler == 'jaxns':
      priors = likelihood_jax.priors
      likelihood = likelihood_jax  
      
 result = bilby.run_sampler(
           likelihood,
           priors=priors,
           sampler=sampler,
           label=sampler + '_' + model,
           outdir=outdir,
           likelihood_benchmark=True,
           resume=False,
           clean=True,
           verbose=True,
           **sampler_dict
          )

 f = open(outdir + '/'+ model + '_' + sampler + '_leval.txt', "w")
 print("=" * 40)
 print(sampler)
 print("=" * 40)
 print(result)
 print('Number of likelihood evaluations', result.num_likelihood_evaluations)
 print("=" * 40)
 f.write(sampler + ':' + str(result.num_likelihood_evaluations) + '\n')

 f.close()

 f = open(outdir + '/'+ model + '_'  + sampler + '_log_evidence.txt', "w")
 print("=" * 40)
 print('Log of evidence', result.log_evidence, '+/-' , result.log_evidence_err)
 print("=" * 40)
 f.write(sampler + ':' + str(result.log_evidence) + '+/-' + str(result.log_evidence_err) +  '\n')

 f.close()

 f = open(outdir + '/'+ model + '_'  + sampler + '_hyper.txt', "w")
 if sampler != 'ultranest' and sampler != 'dynamic_dynesty': 
  print('Number of live points', nlive)
  f.write(sampler + ':{nlive=' + str(nlive) + '}' + '\n')
 if sampler == 'ultranest':
  print('Number of steps', nsteps)
  f.write(sampler + ':{nteps=' + str(nsteps) + '}'+ '\n')
 if sampler == 'dynamic_dynesty' or sample == 'dynesty':
  print('Bound', bound)
  print('Sample', sample)
  print('Number of walks', walks)
  f.write(sampler + ':{bound=' + str(bound) + '}' + '\n') 
  f.write(sampler + ':{sample=' + str(sample) + '}' + '\n') 
  f.write(sampler + ':{walks=' + str(walks) + '}' + '\n')   

 f.close()

 result.plot_corner(filename=outdir + '/' + model + '_' + sampler + '.pdf')

 #plt.savefig(outdir + '/' + model + '_' + sampler + '.pdf')
 #plt.close()
 
 print("--- %s seconds ---" % (time.time() - script_start))
