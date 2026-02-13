import torch
import torch.nn as nn
import numpy as np

from se3dif.utils import SO3_R3
import theseus as th
from theseus import SO3
from icecream import ic

class DualProjectedSE3DenoisingLoss():
    def __init__(self, field='dual_denoise', delta = 1., grad=False):
        self.field = field
        self.delta = delta
        self.grad = grad
        self.bce_loss = nn.BCELoss()

    # TODO check sigma value
    def marginal_prob_std(self, t, sigma=0.25):
        return torch.sqrt((sigma ** (2 * t) - 1.) / (2. * np.log(sigma)))

    def loss_fn(self, model, model_input, ground_truth, val=False, eps=1e-5):

        ## Set input 
        H = model_input['x_ene_pos']
        c = model_input['visual_context']
        batch = H.shape[0]
        model.set_latent(c, batch=H.shape[1])
        H = H.reshape(-1, 4, 4)
        
        ## H to vector 
        H_th = SO3_R3(R=H[...,:3, :3], t=H[...,:3, -1])
        xw = H_th.log_map()
        xw = xw.reshape(-1, 2 * xw.shape[-1])
        
        ## Sample perturbed datapoint 
        random_t = torch.rand_like(xw[...,0], device=xw.device) * (1. - eps) + eps
        z = torch.randn_like(xw)
        std = self.marginal_prob_std(random_t)
        perturbed_x = xw + z * std[..., None]
        perturbed_x = perturbed_x.detach()
        perturbed_x.requires_grad_(True)

        ## Get gradient 
        with torch.set_grad_enabled(True):
            perturbed_H = SO3_R3().exp_map(perturbed_x.reshape(2*perturbed_x.shape[0], -1)).to_matrix()
            energy = model(perturbed_H, random_t.unsqueeze(1).repeat(1,2).reshape(-1), batch=batch, dual=True)
            grad_energy = torch.autograd.grad(energy.sum(), perturbed_x,
                                              only_inputs=True, retain_graph=True, create_graph=True)[0]

        # Calculate the L1 loss of the pred score and gt score 
        z_target = z/std[...,None]
        loss_fn = nn.L1Loss()
        loss = loss_fn(grad_energy, z_target)/10

        info = {self.field: grad_energy}
        loss_dict = {"Dual Score loss": loss }
        return loss_dict, info

