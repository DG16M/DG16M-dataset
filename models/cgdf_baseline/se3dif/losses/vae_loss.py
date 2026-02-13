import torch
import torch.nn as nn
from se3dif.utils import SO3_R3
import theseus as th
from theseus import SO3
from icecream import ic

class VAELoss:
    def __init__(self, field='vae'):
        self.field = field
        self.loss_fxn = nn.MSELoss(reduction='mean')
        
    def loss_fn(self, model, model_input, ground_truth, eps=1e-5):
        
        H = model_input['x_ene_pos']
        c = model_input['visual_context']
        batch = H.shape[0]
        model.set_latent(c, batch=H.shape[1])
        H = H.reshape(-1, 4, 4)
        
        H_th = SO3_R3(R=H[...,:3, :3], t=H[...,:3, -1])
        xw = H_th.log_map()
        xw = xw.reshape(-1, 2 * xw.shape[-1])
        
        random_t = torch.rand_like(xw[...,0], device=xw.device) * (1. - eps) + eps

        perturbed_H = SO3_R3().exp_map(xw.reshape(2*xw.shape[0], -1)).to_matrix()
        recon_x, mu, logvar = model(perturbed_H, 
              random_t.unsqueeze(1).repeat(1, 2).reshape(-1, 1), 
              batch=batch)
        
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = self.loss_fxn(recon_x, xw) + KLD
        info = {'recon_x': recon_x}
        loss_dict = {self.field: loss}
        
        return loss_dict, info
        
        
