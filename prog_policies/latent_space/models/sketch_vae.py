import torch

from .base_vae import ModelReturn
from .leaps_vae import LeapsVAE

class SketchVAE(LeapsVAE):
    
    def forward(self, data_batch: tuple, prog_teacher_enforcing = True,
                a_h_teacher_enforcing = True) -> ModelReturn:
        s_h, a_h, a_h_masks, progs, progs_masks = data_batch

        z = self.encode(progs, progs_masks)
        
        decoder_result = self.decode(z, progs, progs_masks, prog_teacher_enforcing)
        pred_progs, pred_progs_logits, pred_progs_masks = decoder_result
        
        return ModelReturn(pred_progs, pred_progs_logits, pred_progs_masks,
                           None, None, None)
