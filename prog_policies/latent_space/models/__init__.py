import torch

from .base_vae import BaseVAE
from .leaps_vae import LeapsVAE
from .leaps_vq_vae import LeapsVQVAE
from .leaps_perceptions_vae import LeapsPerceptionsVAE
from .sketch_vae import SketchVAE
from .dss_vae import DSSVAE
from .dss_vae2 import DSSVAE2
from .dss_vae_cont import DSSVAECont
from .dss_vae_cont2 import DSSVAECont2
from .dss_vae_cont3 import DSSVAECont3
from .dss_vae_cont4 import DSSVAECont4
from .dss_no_adv_vae import DSSNoAdvVAE
from .dss_no_adv_vae2 import DSSNoAdvVAE2

def load_model(model_cls_name: str, model_args: dict,
               model_params_path: str = None) -> BaseVAE:
    model_cls = globals().get(model_cls_name)
    assert issubclass(model_cls, BaseVAE)
    model = model_cls(**model_args)
    if model_params_path is not None:
        model.load_state_dict(torch.load(model_params_path, map_location=model.device))
    return model
