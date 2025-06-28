import torch
from torch.utils.data import DataLoader
from models.dual_gpd_model import DualGPDClassifier
from datasets.dual_arm_gpd_dataset import DualGraspGpdDataset
from trainers.pointnet_gpd_trainer import PointnetGPDTrainer
import random
import numpy as np
import argparse
import os
from omegaconf import OmegaConf

def seed_all(seed=28):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
        
def count_params(model):
    return sum([p.numel() for p in model.parameters() if p.requires_grad])

def combine_args(args, custom_args_path):
    new_args = {}
    for key, value in vars(args).items():
        new_args[key] = value
    custom_args = OmegaConf.load(custom_args_path)
    
    for key, value in custom_args.items():
        new_args[key] = value
    return new_args

def get_time_and_date():
    import datetime
    now = datetime.datetime.now()
    return now.strftime("%d-%m-%H:%M:%S")

def create_experiment_folder(args):
    folder_name = os.path.join(args['exp_dir'], get_time_and_date())
    os.makedirs(folder_name)
    return folder_name
    

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--name', type=str, required=False)
    args = parser.parse_args()
    
    args = combine_args(args, args.config)
    seed_all(args['seed'])
    args['exp_dir'] = create_experiment_folder(args)
    
    if args['save_model_path'] is not None:
        args['save_model_path'] = os.path.join(args['exp_dir'], args['save_model_path'])
        os.makedirs(args['save_model_path'])
    
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    train_meshes = open(args['train_meshes'], 'r').readlines()
    test_meshes = open(args['test_meshes'], 'r').readlines()
    MESH_PATH = args['mesh_path']
    GRASP_PATH = args['grasp_path']
    
    for train_mesh in train_meshes:
        train_mesh = train_mesh.strip()
        if train_mesh.split('.')[-1] != 'obj':
            print(train_mesh)
    
    train_dataset = DualGraspGpdDataset(mesh_path=MESH_PATH, 
                                            grasp_path=GRASP_PATH,
                                            meshes_to_take=train_meshes,
                                            is_training=True,
                                            args=args)
    
    val_dataset = DualGraspGpdDataset(mesh_path=MESH_PATH, 
                                            grasp_path=GRASP_PATH,
                                            meshes_to_take=test_meshes,
                                            is_training=False,
                                            args=args)
    
    train_loader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True, num_workers=10, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args['batch_size'], shuffle=False, num_workers=10)
    
    
    data = next(iter(val_loader))
    torch.save(data, 'example_val_batch.pt')
    model = DualGPDClassifier()
    print(f'Parameters: {count_params(model)}')
    
    trainer = PointnetGPDTrainer(model=model,
                                 train_loader=train_loader,
                                 val_loader=val_loader,
                                 device=device,
                                 args=args)
    
    trainer.train()
    
if __name__ == "__main__":
    main()
