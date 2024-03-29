from jaxns_cosmology.jaxns_sampler import Jaxns
from jaxns_cosmology.nautilus_sampler import Nautilus
from jaxns_cosmology.ultranest_sampler import Ultranest


# from bilby.core.sampler import IMPLEMENTED_SAMPLERS

def install_jaxns_sampler():
    from bilby.core.sampler import IMPLEMENTED_SAMPLERS
    global IMPLEMENTED_SAMPLERS
    IMPLEMENTED_SAMPLERS['jaxns'] = Jaxns


def install_nautilus_sampler():
    from bilby.core.sampler import IMPLEMENTED_SAMPLERS
    global IMPLEMENTED_SAMPLERS
    IMPLEMENTED_SAMPLERS['nautilus'] = Nautilus


def install_ultranest_sampler():
    from bilby.core.sampler import IMPLEMENTED_SAMPLERS
    global IMPLEMENTED_SAMPLERS
    IMPLEMENTED_SAMPLERS['ultranest'] = Ultranest
