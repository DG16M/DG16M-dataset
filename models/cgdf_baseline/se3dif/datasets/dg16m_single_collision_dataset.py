import os
import torch
import torch.nn as nn
import h5py
from torch.utils.data import Dataset
import trimesh
from tqdm import tqdm
import random 
from scipy.spatial.transform import Rotation as R   
import numpy as np
from trimesh.collision import CollisionManager


class SingleCollisionGraspDG16M(Dataset):
    def __init__(self, mesh_path, grasp_path, meshes_to_take, args, is_training=True,
                 train_generation_model=False):
        super().__init__()
        self.mesh_path = mesh_path
        self.grasp_path = grasp_path
        self.args = args
        self.meshes_to_take = meshes_to_take    
        self.mesh_files = [os.path.join(mesh_path, i) for i in self.meshes_to_take]
        random.shuffle(self.mesh_files)
        
        self.contact_points_to_take = 48
        
        self.objects = {}
        self.train_generation_model = train_generation_model  
        
        for mesh_file in tqdm(self.mesh_files):
            mesh_name = mesh_file.strip()
            obj = trimesh.load(mesh_file.strip())
            grasp_file = os.path.join(grasp_path, mesh_name.split("/")[-1].replace(".obj", ".h5"))  
            grasp_file = h5py.File(grasp_file, 'r') 
            
            grasps = grasp_file['grasps/grasps'][()].reshape(-1, 4, 4)
            # grasps = np.unique(grasps, axis=0)
            
            self.objects[mesh_name] = {
                'mesh': obj,
                'grasps': grasps
            }
            
        self.mesh_files = list(self.objects.keys())
        self.n_samples = len(self.objects)
        self.is_training = is_training
        
        self.gripper = trimesh.load('/home/dualarm/dummyhome/md/March2025/dual-arm-grasp-diffusion/gripper.obj')
        print('Number of objects:', self.n_samples)
        
    def __len__(self):
        return self.n_samples
    
    def create_random_T(self):
        T = np.eye(4)
        T[:3, :3] = R.from_euler('xyz', np.random.randn(3) * 30, degrees=True).as_matrix()
        T[:3, 3] = np.random.uniform(-0.03, 0.03, size=3)
        return T
    
    def collision_checker(self, mesh, grasp_left=None, grasp_right=None):
        cm = CollisionManager()
        cm.add_object('mesh', mesh)
        if grasp_left is not None:
            cm.add_object('gripper_left', grasp_left)
        if grasp_right is not None:
            cm.add_object('gripper_right', grasp_right)
        return cm.in_collision_internal()
       
    def get_k_nearest(self, pcd, point, k=512):
        dists = np.linalg.norm(pcd - point[None, :], axis=1)
        mask = np.argsort(dists)[:k]
        return pcd[mask]
    
    def __getitem__(self, idx):
        mesh_name = self.mesh_files[idx].strip()
        label = 1
        
        
        obj = self.objects[mesh_name]['mesh']
        grasp_og = self.objects[mesh_name]['grasps']
        i = np.random.randint(0, len(grasp_og))
        grasp = grasp_og[i].copy()
        
        if np.random.rand(1) > 0.25:
            # grasp = grasp @ self.create_random_T()
            random_T = self.create_random_T()
            grasp[:3, :3] = grasp[:3, :3] @ random_T[:3, :3]
            grasp[:3, 3] = grasp[:3, 3] + random_T[:3, 3]
            gripper_now = self.gripper.copy().apply_transform(grasp)
            if self.collision_checker(obj, grasp_left=gripper_now, grasp_right=None):
                label = 0
            else:
                grasp = grasp_og[i].copy()
        
        pcd = obj.sample(5000)
        pcd_part = self.get_k_nearest(pcd, grasp[:3, 3], k=512)
        # print(pcd_part.shape, label)
        # exit()
        
        return {
            'pcd': torch.from_numpy(pcd_part).float(),
            'grasp': torch.from_numpy(grasp).float(),
            'label': torch.tensor(label).float(),
            'object_name': mesh_name
        }
          

            
            
            
            
            
            