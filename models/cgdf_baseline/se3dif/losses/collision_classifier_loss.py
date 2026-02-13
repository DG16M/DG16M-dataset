import torch
import torch.nn as nn
from se3dif.utils import SO3_R3
import numpy as np

class CollisionClassifierLoss2():
    def __init__(self, field='classifier', delta = 0.6, grad=True):
        self.field = field
        self.delta = delta

        self.grad = grad
        self.bce_loss = nn.BCELoss()
        self.GRAD_LOSS_COEFF = 0.01 

    def loss_fn(self, model, model_input, ground_truth, val=False, eps=1e-5):
        
        H = model_input['x_ene_pos'] # bs, 150, 2, 4, 4
        c = model_input['visual_context'] # bs, 1000, 3
        batch = H.shape[0] # bs
        model.set_latent(c, batch=H.shape[1])
        H = H.reshape(-1, 4, 4) # bs*150*2, 4, 4
        
        H_th = SO3_R3(R=H[...,:3, :3], t=H[...,:3, -1])
        xw = H_th.log_map() # bs*150*2, 6
        # xw = xw.reshape(-1, 2 * xw.shape[-1]) # bs*150, 12
        # print(xw.shape)
        ## 2. Sample perturbed datapoint ##
        random_t = torch.rand_like(xw[...,0], device=xw.device) * (1. - eps) + eps
        
        random_t = random_t * 0
        
        perturbed_x = xw + 0

        perturbed_H = SO3_R3().exp_map(perturbed_x.reshape(perturbed_x.shape[0], -1)).to_matrix()

        print(perturbed_H)
        pred = model(perturbed_H, random_t, 
                       batch=batch, dual=True, collision_forward=True)

        # print("Percentage of 0s:", 1 - (ground_truth['labels'].mean()))
        loss = self.bce_loss(pred.reshape(-1), ground_truth['labels'].reshape(-1))/5
        acc = torch.sum((pred.reshape(-1) > 0.5).float() == ground_truth['labels'].reshape(-1)).item() / ground_truth['labels'].numel()
        
        # logits = torch.log(pred / (1 - pred + eps))
        # logits_scalar = logits.mean()
        # grads = torch.autograd.grad(
        #     outputs=logits_scalar, 
        #     inputs=[p for p in model.parameters()], 
        #     create_graph=True,  # Important if you want to backprop through grad_norm
        #     retain_graph=True   # Retain for further backward (like loss)
        # )

        # grad_norm = torch.norm(torch.stack([
        #     g.norm(2) for g in grads if g is not None
        # ]))
        # grad_loss = (grad_norm - 1) ** 2
        
        # loss = loss + self.GRAD_LOSS_COEFF * grad_loss
        
        info = {self.field: pred.reshape(-1)}
        loss_dict = {"Collision Classifier Loss": loss,
                     'Collision Classifier Accuracy': acc}
        
        return loss_dict, info
        
class CollisionClassifierLoss():
    def __init__(self, field='classifier', delta = 0.6, grad=True):
        self.field = field
        self.delta = delta

        self.grad = grad
        self.bce_loss = nn.BCELoss()
        self.GRAD_LOSS_COEFF = 0.01 
        
    def marginal_prob_std(self, t, sigma=0.25):
        return torch.sqrt((sigma ** (2 * t) - 1.) / (2. * np.log(sigma)))

    def loss_fn(self, model, model_input, ground_truth, val=False, eps=1e-5):
        
        H = model_input['x_ene_pos'].detach() # bs, 150, 2, 4, 4
        c = model_input['visual_context'].detach() # bs, 1000, 3
        batch = H.shape[0] # bs
        model.set_latent(c, batch=H.shape[1])
        H = H.reshape(-1, 4, 4) # bs*150*2, 4, 4
        
        H_th = SO3_R3(R=H[...,:3, :3], t=H[...,:3, -1])
        xw = H_th.log_map() # bs*150*2, 6
        perturbed_x = xw.clone() # bs*150*2, 6
        
        n = len(xw)
        gt = ground_truth['labels'].reshape(-1)
        
        neg_indices_to_perturb = torch.where(gt == 0)[0]
        neg_indices_to_perturb = neg_indices_to_perturb[torch.randperm(len(neg_indices_to_perturb))[:n//8]]
        # neg_indices_to_perturb = torch.randperm(len(neg_indices_to_perturb))[:n//2]
        
        random_t = 2.0 * torch.rand(len(neg_indices_to_perturb), device=xw.device)
        z = torch.randn_like(xw[neg_indices_to_perturb])
        std = self.marginal_prob_std(random_t)
                    
        for j in [3, 5]:
            z[:, j] = z[:, j] + 1.5
        
        z[:, 4] = z[:, 4] * 1e-1
        perturbed_x[neg_indices_to_perturb] = xw[neg_indices_to_perturb] + z * std[..., None]
        perturbed_H = SO3_R3().exp_map(perturbed_x.reshape(perturbed_x.shape[0], -1)).to_matrix()
        
        # torch.save(perturbed_H, './temp/perturbed_H.pth')
        # torch.save(c, './temp/c.pth')
        # torch.save(gt, './temp/gt.pth')
        # exit()
        
        # print('Saved')
        # exit()
        
        # n_grasps = n//len(c)
        # gt = torch.zeros(len(c), n_grasps).to(xw.device)
        # perturbed_x = torch.zeros(len(c), n_grasps, 6).to(xw.device)
        # xw = xw.reshape(len(c), n_grasps, -1)
        
        # for i in range(len(c)):            
        #     low_half = 0.001 * torch.rand(n_grasps//2, device=xw.device) # good grasps
        #     high_half = 1.0 * torch.rand(n_grasps//2, device=xw.device) # bad grasps
        #     random_t = torch.cat([low_half, high_half], dim=0)
            
        #     gt_now = torch.cat([
        #         torch.ones(n_grasps//2), 
        #         torch.zeros(n_grasps//2)
        #     ]).to(xw.device)
            
        #     z = torch.randn_like(perturbed_x[0, ...])
        #     std = self.marginal_prob_std(random_t)
        #     for j in [0, 1, 2]:
        #         z[n_grasps//2:, j] = z[n_grasps//2:, j] * (0.03 if np.random.rand(1) > 0.5 else 1)
                
        #     for j in [3, 5]:
        #         z[n_grasps//2:, j] = z[n_grasps//2:, j] + 1.5
                
        #     z[n_grasps//2:, 4] = z[n_grasps//2:, 4] * 0
            
        #     perturbed_x_now = xw[i] + z * std[..., None]
        #     perm = torch.randperm(n_grasps, device=xw.device)
        #     perturbed_x[i] = perturbed_x_now[perm].clone()
        #     gt[i, :] = gt_now[perm].clone()
        #     # print(perm.shape, gt[i])
        
        # perturbed_x = perturbed_x.reshape(n, -1)
        # gt = gt.reshape(n, -1)
        
        # perturbed_H = SO3_R3().exp_map(perturbed_x.reshape(perturbed_x.shape[0], -1)).to_matrix()
        
        random_t = torch.rand(n).to(xw.device)
        
        pred = model(perturbed_H, random_t * 1e-3,
                       batch=batch, dual=True, collision_forward=True)
        
        loss = self.bce_loss(pred.reshape(-1), gt.reshape(-1))/5
        # print(pred)
        acc = torch.sum((pred.reshape(-1) > 0.5).float() == gt.reshape(-1)).item() / gt.numel()
        
        info = {self.field: pred.reshape(-1)}
        loss_dict = {"Collision Classifier Loss": loss,
                     'Collision Classifier Accuracy': acc}
        
        return loss_dict, info


class CollisionClassifierLoss3():
    def __init__(self, field='classifier', delta = 0.6, grad=True):
        self.field = field
        self.delta = delta

        self.grad = grad
        self.bce_loss = nn.BCELoss()
        self.GRAD_LOSS_COEFF = 0.01 
        
    def marginal_prob_std(self, t, sigma=0.25):
        return torch.sqrt((sigma ** (2 * t) - 1.) / (2. * np.log(sigma)))

    def loss_fn(self, model, model_input, ground_truth, val=False, eps=1e-5):
        
        H = model_input['x_ene_pos'].detach() # bs, 150, 2, 4, 4
        c = model_input['visual_context'].detach() # bs, 1000, 3
        batch = H.shape[0] # bs
        model.set_latent(c, batch=H.shape[1])
        H = H.reshape(-1, 4, 4) # bs*150*2, 4, 4
        
        H_th = SO3_R3(R=H[...,:3, :3], t=H[...,:3, -1])
        xw = H_th.log_map() # bs*150*2, 6
        # xw = xw.reshape(-1, 2 * xw.shape[-1]) # bs*150, 12
        # print(xw.shape)
        ## 2. Sample perturbed datapoint ##
        # random_t = torch.rand_like(xw[...,0], device=xw.device) * (1. - eps) + eps
        # print(random_t)
        n = len(xw)
        n_grasps = n//len(c)
        gt = torch.zeros(len(c), n_grasps).to(xw.device)
        perturbed_x = torch.zeros(len(c), n_grasps, 6).to(xw.device)
        xw = xw.reshape(len(c), n_grasps, -1)
        
        for i in range(len(c)):            
            low_half = 0.001 * torch.rand(n_grasps//2, device=xw.device) # good grasps
            high_half = 1.0 * torch.rand(n_grasps//2, device=xw.device) # bad grasps
            random_t = torch.cat([low_half, high_half], dim=0)
            
            gt_now = torch.cat([
                torch.ones(n_grasps//2), 
                torch.zeros(n_grasps//2)
            ]).to(xw.device)
            
            z = torch.randn_like(perturbed_x[0, ...])
            std = self.marginal_prob_std(random_t)
            for j in [0, 1, 2]:
                z[n_grasps//2:, j] = z[n_grasps//2:, j] * (0.03 if np.random.rand(1) > 0.5 else 1)
                
            for j in [3, 5]:
                z[n_grasps//2:, j] = z[n_grasps//2:, j] + 1.5
                
            z[n_grasps//2:, 4] = z[n_grasps//2:, 4] * 0
            
            perturbed_x_now = xw[i] + z * std[..., None]
            perm = torch.randperm(n_grasps, device=xw.device)
            perturbed_x[i] = perturbed_x_now[perm].clone()
            gt[i, :] = gt_now[perm].clone()
            # print(perm.shape, gt[i])
        
        perturbed_x = perturbed_x.reshape(n, -1)
        gt = gt.reshape(n, -1)
        
        perturbed_H = SO3_R3().exp_map(perturbed_x.reshape(perturbed_x.shape[0], -1)).to_matrix()
        
        random_t = torch.zeros(n).to(xw.device)
        
        pred = model(perturbed_H, random_t * 1e-3,
                       batch=batch, dual=True, collision_forward=True)
        
        loss = self.bce_loss(pred.reshape(-1), gt.reshape(-1))/5
        # print(pred)
        acc = torch.sum((pred.reshape(-1) > 0.5).float() == gt.reshape(-1)).item() / gt.numel()
        
        # logits = torch.log(pred / (1 - pred + eps))
        # logits_scalar = logits.mean()
        # grads = torch.autograd.grad(
        #     outputs=logits_scalar, 
        #     inputs=[p for p in model.parameters()], 
        #     create_graph=True,  # Important if you want to backprop through grad_norm
        #     retain_graph=True   # Retain for further backward (like loss)
        # )

        # grad_norm = torch.norm(torch.stack([
        #     g.norm(2) for g in grads if g is not None
        # ]))
        # grad_loss = (grad_norm - 1) ** 2
        
        # loss = loss + self.GRAD_LOSS_COEFF * grad_loss
        
        info = {self.field: pred.reshape(-1)}
        loss_dict = {"Collision Classifier Loss": loss,
                     'Collision Classifier Accuracy': acc}
        
        return loss_dict, info
