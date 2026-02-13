from models.cgdf_baseline.se3dif.datasets.dg16m_dataset import DG16MPointcloudSDFDataset
from torch.utils.data import DataLoader
from models.cgdf_baseline.se3dif.utils import load_experiment_specifications
from models.cgdf_baseline.se3dif import trainer
from models.cgdf_baseline.se3dif.losses.main import get_losses
from models.cgdf_baseline.se3dif.models import loader
import torch
import torch.optim as optim
import os
import numpy as np
import random
from torchinfo import summary as model_summary
import argparse 

def seed_all(seed=28):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to the config file.')
    args_cmd = parser.parse_args()

    spec_file = './'
    args = load_experiment_specifications(spec_file, load_yaml=args_cmd.config)
    
    seed_all(seed=args['seed'])

    grasps_dir = args['grasps_dir']
    meshes_dir = args['meshes_dir']
    sdf_dir = args['sdf_dir']   
    
    train_meshes_to_take = open(args['train_meshes_list']).readlines()
    train_meshes_to_take = [mesh.strip() for mesh in train_meshes_to_take]
    
    val_meshes_to_take = open(args['val_meshes_list']).readlines()
    val_meshes_to_take = [mesh.strip() for mesh in val_meshes_to_take]

    print('Creating Datasets and Dataloaders...')
    
    train_dataset = DG16MPointcloudSDFDataset(
        grasps_dir=grasps_dir,
        meshes_dir=meshes_dir,
        sdf_dir=sdf_dir,
        single_arm=args['single_arm'],
        meshes_to_take=train_meshes_to_take,
        n_points = args['num_input_points'],
        n_grasps=args['num_input_grasps']
    )
    
    val_dataset = DG16MPointcloudSDFDataset(
        grasps_dir=grasps_dir,
        meshes_dir=meshes_dir,
        sdf_dir=sdf_dir,
        single_arm=args['single_arm'],
        meshes_to_take=val_meshes_to_take,
        n_points = args['num_input_points'],
        n_grasps=args['num_input_grasps'],
    )
    
    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=args['TrainSpecs']['batch_size'], 
                                  shuffle=True, 
                                  num_workers=args['TrainSpecs']['num_workers'])
    val_loader = DataLoader(val_dataset,
                            batch_size=args['TrainSpecs']['val_batch_size'], 
                            shuffle=False, 
                            num_workers=args['TrainSpecs']['num_workers'])
    print('Datasets and Dataloaders Created!')
    
    exp_dir = os.path.join('.', args['exp_log_dir'])
    args['saving_folder'] = exp_dir
    
    res, gt = next(iter(train_dataloader))
    
    print("Sample Input Batch:")
    for k, v in res.items():
        print(k, v.shape)
    print('=' * 50)
    print("Ground Truth:")
    for k, v in gt.items():
        print(k, v.shape)
    
    device = 'cuda'
    args['device'] = device
    
    print("Loading Model, Losses, and Optimizer...")
    model = loader.load_model(args)
    
    loss = get_losses(args)
    loss_fn = val_loss_fn = loss.loss_fn
    lr = args['learning_rate']
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print('Number of parameters:', count_params(model))
    
    model_summary(model, col_names=['num_params', 'trainable'])

    print("Starting Training...")   
    trainer.train(
        model=model.float(), 
        train_dataloader=train_dataloader, 
        epochs=args['TrainSpecs']['num_epochs'], 
        device=device, 
        optimizers=[optimizer],
        steps_til_summary=args['TrainSpecs']['steps_til_summary'],
        epochs_til_checkpoint=args['TrainSpecs']['epochs_til_checkpoint'],
        model_dir= exp_dir, 
        loss_fn=loss_fn, 
        clip_grad=False, 
        val_loss_fn=val_loss_fn,
        val_dataloader=val_loader, 
        args=args
    )

    print("Training Complete!")

    
if __name__ == '__main__':
    main()