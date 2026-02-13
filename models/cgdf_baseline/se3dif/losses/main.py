from .denoising_loss import DualProjectedSE3DenoisingLoss
from .collision_classifier_loss import CollisionClassifierLoss
from .sdf_loss import SDFLoss
from .classifier_loss import ClassifierLoss
from .vae_loss import VAELoss

def get_losses(args):
    losses = args['Losses']

    loss_fns = {}
    if 'sdf_loss' in losses:
        loss_fns['sdf'] = SDFLoss().loss_fn

    if 'dual_projected_denoising_loss' in losses:
        loss_fns['denoise'] = DualProjectedSE3DenoisingLoss().loss_fn

    if 'classifier_loss' in losses:
        loss_fns['classifier'] = ClassifierLoss().loss_fn

    if 'vae_loss' in losses:
        loss_fns['vae'] = VAELoss().loss_fn

    if 'collision_classifier_loss' in losses:
        loss_fns['collision_classifier'] = CollisionClassifierLoss().loss_fn
        
    loss_dict = LossDictionary(loss_dict=loss_fns)
    return loss_dict


class LossDictionary():

    def __init__(self, loss_dict):
        self.fields = loss_dict.keys()
        self.loss_dict = loss_dict

    def loss_fn(self, model, model_input, ground_truth, val=False):

        losses = {}
        infos = {}
        for field in self.fields:
            loss_fn_k = self.loss_dict[field]
            loss, info = loss_fn_k(model, model_input, ground_truth, val)
            losses = {**losses, **loss}
            infos = {**infos, **info}

        return losses, infos