from typing import Dict

from jaxns import Model

from jaxns_cosmology.models.CMB import build_CMB_model
from jaxns_cosmology.models.MSSM7 import build_MSSM7_model
from jaxns_cosmology.models.eggbox import build_eggbox_model
from jaxns_cosmology.models.rastrigin import build_rastrigin_model
from jaxns_cosmology.models.rosenbrock import build_rosenbrock_model
from jaxns_cosmology.models.spikeslab import build_spikeslab_model


def all_models() -> Dict[str, Model]:
    """
    Return all the models

    Returns:
        A dictionary of models
    """
    return dict(
        CMB=build_CMB_model(),
        MSSM7=build_MSSM7_model(),
        eggbox=build_eggbox_model(ndim=10),
        rastrigin=build_rastrigin_model(ndim=10),
        rosenbrock=build_rosenbrock_model(ndim=10),
        spikeslab=build_spikeslab_model(ndim=10)
    )
