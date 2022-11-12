from jaxns_cosmology.jaxns_sampler import Jaxns
# from bilby.core.sampler import IMPLEMENTED_SAMPLERS

def install_jaxns_sampler():
    from bilby.core.sampler import IMPLEMENTED_SAMPLERS
    global IMPLEMENTED_SAMPLERS
    IMPLEMENTED_SAMPLERS['jaxns'] = Jaxns