from icecream import ic
import torch
import torch.nn as nn
from se3dif.utils import SO3_R3


class ClassifierLoss():
    def __init__(self):
        self.bce_loss = nn.BCELoss()

    def compute_acc(self, pred, labels):
        return torch.sum((pred > 0.5).float() == labels).item() / labels.numel()

    def loss_fn(self, model, model_input, ground_truth, val=False, eps=1e-5):
        
        H = model_input['x_ene_pos']
        c = model_input['visual_context']
        gt_labels = ground_truth['labels']

        batch = H.shape[0]
        model.set_latent(c, batch=H.shape[1])
        H = H.reshape(-1, 4, 4)

        H_th = SO3_R3(R=H[...,:3, :3], t=H[...,:3, -1])
        xw = H_th.log_map()
        xw = xw.reshape(-1, 2 * xw.shape[-1])

        t = torch.zeros_like(xw[...,0], device=xw.device)
        
        H_input = SO3_R3().exp_map(xw.reshape(2*xw.shape[0], -1)).to_matrix()
        fc_pred = model(H = H_input, 
                        k = t.unsqueeze(1).repeat(1,2).reshape(-1), 
                        batch=batch)

        classifier_loss = self.bce_loss(fc_pred.reshape(-1), gt_labels.reshape(-1))

        with torch.no_grad():
            acc = self.compute_acc(fc_pred.reshape(-1), gt_labels.reshape(-1))
            
        info = {}
        loss_dict = {
            "Classifier Loss": classifier_loss,
            "Classifier Accuracy": acc
        }

        return loss_dict, info