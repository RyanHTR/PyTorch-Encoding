from .model_zoo import get_model
from .model_store import get_model_file
from .resnet import *
from .cifarresnet import *
from .base import *
from .fcn import *
from .psp import *
from .encnet import *
from .deeplab import *
from .multi_nl_fcn import *


def get_segmentation_model(name, **kwargs):
    from .fcn import get_fcn
    models = {
        'fcn': get_fcn,
        'psp': get_psp,
        'encnet': get_encnet,
        'deeplab': get_deeplab,
        'multi_nl_fcn': get_multi_nl_fcn,
    }
    return models[name.lower()](**kwargs)
