import glob
import copy
import time

import numpy as np
import trimesh

from scipy.stats import special_ortho_group

import os
import torch

from torch.utils.data import DataLoader, Dataset
import json
import pickle
import h5py
from se3dif.utils import get_data_src

from se3dif.utils import to_numpy, to_torch, get_grasps_src
# from mesh_to_sdf.surface_point_cloud import get_scan_view, get_hq_scan_view
# from mesh_to_sdf.scan import ScanPointcloud
from tqdm import tqdm
from icecream import ic


import os, sys

import logging
logger = logging.getLogger("trimesh")
logger.setLevel(logging.ERROR)
from scipy.spatial.transform import Rotation as R


class DA2grasps:
    def __init__(self, grasp_path, mesh_path, scales, single_arm=False):
        self.grasp_path = grasp_path
        self.mesh_path = mesh_path
        data = h5py.File(self.grasp_path, 'r')
        self.mesh_name = self.mesh_path.split('/')[-1].split('.')[0]
        self.mesh_scale = scales[self.mesh_name]
        self.single_arm = single_arm
        self.positive_negative = True
        
        self.grasps = self.load_grasps(data)
        self.mesh = self.load_mesh()        
            
    def load_grasps(self, data):
        grasps = data['grasps/transforms'][()]
        if self.single_arm:
            return np.unique(grasps.reshape(-1, 4, 4), axis=0) 
        
        fc = data['grasps/qualities/Force_closure'][:]
        dex = data['grasps/qualities/Dexterity'][:]
        tor = data['grasps/qualities/Torque_optimization'][:]

        quality = 0.4 * fc + 0.5 * dex + 0.1 * tor
        
        positive_indices = np.where(quality >= 0.92)[0]
        negative_indices = np.where(quality <= 0.85)[0]
        
        positive_grasps = grasps[positive_indices]
        negative_grasps = grasps[negative_indices]
        if self.positive_negative: 
            grasps = np.concatenate((positive_grasps, negative_grasps), axis=0)
            self.labels = np.concatenate((np.ones(positive_grasps.shape[0]), 
                                 np.zeros(negative_grasps.shape[0])), axis=0)
        else:
            grasps = grasps[positive_indices]
            self.labels = np.ones(positive_grasps.shape[0])
        return grasps
    
    def load_mesh(self):
        mesh = trimesh.load(self.mesh_path)
        # mesh.apply_translation(-mesh.centroid)
        mesh.apply_scale(self.mesh_scale)
        return mesh
    
class DA2PointcloudSDFDataset(Dataset):
    def __init__(self, 
                 grasps_dir, 
                 meshes_dir, 
                 sdf_dir, 
                 meshes_to_take,
                 n_grasps=64, 
                 n_points=1000,
                 single_arm=False):
        super().__init__()
        self.n_grasps = n_grasps
        self.n_points = n_points
        self.grasps_dir = grasps_dir
        self.meshes_dir = meshes_dir
        self.sdf_dir = sdf_dir
        self.single_arm = single_arm
        self.meshes_to_take = meshes_to_take        
        # print(self.grasps_dir)
        
        # self.grasp_files = os.listdir(self.grasps_dir)
        # self.scales = json.load('./scales.json')
        with open('./scales.json', 'r') as f:
            self.scales = json.load(f)
        
        self.grasp_files = [i.replace('.obj', '') for i in self.meshes_to_take]
        self.grasp_files = [f"{i}_{self.scales[i]}.h5" for i in self.grasp_files]
        
        self.grasp_objects = []
        
        for grasp_file in tqdm(self.grasp_files, desc='Loading grasps'):
            if not grasp_file.endswith('.h5'):
                continue
            try:
                grasp_path = os.path.join(self.grasps_dir, grasp_file)
                mesh_name = grasp_file.split('_')[0] + '.obj'
                mesh_path = os.path.join(self.meshes_dir, mesh_name)
                # print(grasp_path, mesh_path)
                grasp_object = DA2grasps(grasp_path, mesh_path, self.scales, self.single_arm)
                if grasp_object.grasps.shape[0] > 0:
                    self.grasp_objects.append(grasp_object)
            except Exception as e:
                # print(f'Error loading {grasp_file} | {e}')
                continue
                                    
        np.random.shuffle(self.grasp_objects)
        
        train_size = int(0.95 * len(self.grasp_objects))
        test_size = len(self.grasp_objects) - train_size
        
        self.train_grasp_objects, self.val_grasp_objects = torch.utils.data.random_split(self.grasp_objects, [train_size, test_size])
        self.n_samples = len(self.train_grasp_objects)
        self.custom_scale = 8.0
        
        print(f'Loaded {len(self.grasp_objects)} grasp objects')
        
    def get_sdf(self, grasp_object):
        sdf_path = os.path.join(self.sdf_dir, grasp_object.mesh_name + '.json')
        with open(sdf_path, 'rb') as f:
            sdf_dict = pickle.load(f)
        
        loc = sdf_dict['loc']
        scale = sdf_dict['scale']
        sdf_points = (sdf_dict['xyz'] + loc) * scale * grasp_object.mesh_scale
        indices_to_take = np.random.choice(sdf_points.shape[0], self.n_points, replace=False)
        sdf_points = sdf_points[indices_to_take]
        sdf = sdf_dict['sdf'][indices_to_take] * scale * grasp_object.mesh_scale
        return sdf_points, sdf
        
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        grasp_object = self.grasp_objects[idx]
        pcd = grasp_object.mesh.sample(self.n_points) # [n_points, 3]
        grasps = grasp_object.grasps
        indices_to_take = np.random.choice(grasps.shape[0], self.n_grasps, replace=not grasps.shape[0] > self.n_grasps)
        grasps = grasps[indices_to_take] # [n_grasps, 2, 4, 4]
        labels = grasp_object.labels[indices_to_take] # [n_grasps]
        
        sdf_points, sdf = self.get_sdf(grasp_object)
        
        # scale everything by self.custom_scale
        pcd = pcd * self.custom_scale
        sdf_points = sdf_points * self.custom_scale
        sdf = sdf * self.custom_scale 
        grasps[..., :3, -1] = grasps[..., :3, -1] * self.custom_scale
        
        pcd_mean = np.mean(pcd, axis=0)
        pcd = pcd - pcd_mean
        sdf_points = sdf_points - pcd_mean
        # pcd = pcd - sdf_mean
        
        # random_R = special_ortho_group.rvs(3)
        random_R = R.from_euler('z', np.random.uniform(0, 2 * np.pi), degrees=False).as_matrix()
        # random_R = np.eye(3)
        random_T = np.eye(4)
        random_T[:3, :3] = random_R
        
        pcd = np.einsum('mn,bn->bm',random_R, pcd)
        sdf_points = np.einsum('mn,bn->bm',random_R, sdf_points)
        if self.single_arm:
            grasps = np.einsum('mn,bnk->bmk', random_T, grasps)
        else: 
            grasps = np.einsum('mn,bonk->bomk', random_T, grasps)
            
        # if self.single_arm:
        #     H_grasps = np.einsum('mn,bnk->bmk', H, H_grasps)
        # else:
        #     H_grasps = np.einsum('mn,bonk->bomk', H, H_grasps)
        
        
        res = {
            'visual_context': torch.from_numpy(pcd).float(),
            'x_sdf': torch.from_numpy(sdf_points).float(),
            'scale': torch.Tensor([self.custom_scale]).float(),
            'x_ene_pos': torch.from_numpy(grasps).float()
        }
        
        gt = {
            'sdf': torch.from_numpy(sdf).float(),
            'labels': torch.from_numpy(labels).float(),
        }
        
        return res, gt
