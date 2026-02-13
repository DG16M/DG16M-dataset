import os
import torch
import torch.nn as nn
import h5py
from torch.utils.data import Dataset, DataLoader
import trimesh
from tqdm import tqdm
import random 
from scipy.spatial.transform import Rotation as R   
import numpy as np

class ContactPointClassificationDG16M(Dataset):
    def __init__(self, mesh_path, grasp_path, meshes_to_take, args, is_training=True):
        super().__init__()
        self.mesh_path = mesh_path
        self.grasp_path = grasp_path
        self.args = args
        self.meshes_to_take = meshes_to_take    
        self.mesh_files = [os.path.join(mesh_path, i) for i in self.meshes_to_take]
        random.shuffle(self.mesh_files)
        
        self.contact_points_to_take = 48
        
        self.objects = {}
        
        for mesh_file in tqdm(self.mesh_files):
            mesh_name = mesh_file.strip()
            obj = trimesh.load(mesh_file.strip())
            grasp_file = os.path.join(grasp_path, mesh_name.split("/")[-1].replace(".obj", ".h5"))  
            grasp_file = h5py.File(grasp_file, 'r') 
            contact_points = grasp_file['grasps/contact_points'][()]
            passed_indices = grasp_file['grasps/fc_passing_indices'][()]
            failed_indices = grasp_file['grasps/fc_failed_indices'][()]
            
            if len(passed_indices) < self.contact_points_to_take or len(failed_indices) < self.contact_points_to_take:
                # print(f'No good grasps for {mesh_file}')
                continue
                
            self.objects[mesh_name] = {
                'mesh': obj,
                'positive_contact_points': contact_points[passed_indices],
                'negative_contact_points': contact_points[failed_indices],
            }
            
        self.mesh_files = list(self.objects.keys())
        self.n_samples = len(self.objects)
        self.is_training = is_training  
        
        print('Number of objects:', self.n_samples)
        
    def __len__(self):
        return self.n_samples   
    
    
    def __getitem__(self, idx):
        mesh_name = self.mesh_files[idx].strip()
        obj = self.objects[mesh_name]['mesh']
        positive_contact_points = self.objects[mesh_name]['positive_contact_points']
        negative_contact_points = self.objects[mesh_name]['negative_contact_points']
        
        n_positive = self.contact_points_to_take if len(positive_contact_points) > self.contact_points_to_take else len(positive_contact_points)
        n_negative = self.contact_points_to_take if len(negative_contact_points) > self.contact_points_to_take else len(negative_contact_points)
        
        random_indices = np.random.choice(len(positive_contact_points), n_positive, replace=False)
        positive_contact_points = positive_contact_points[random_indices]
        random_indices = np.random.choice(len(negative_contact_points), n_negative, replace=False)
        negative_contact_points = negative_contact_points[random_indices]
        
        if self.args['random_rotation']:
            random_R = np.random.uniform(0, 2 * np.pi)
            random_R = R.from_euler('z', random_R, degrees=False).as_matrix()
            random_H = np.eye(4)
            random_H[:3, :3] = random_R
            
            obj.apply_transform(random_H)   
            
            # N, 4, 3
            positive_contact_points = np.matmul(positive_contact_points, random_R.T)
            negative_contact_points = np.matmul(negative_contact_points, random_R.T)
                
        pcd = obj.sample(1024)
        labels = [1] * len(positive_contact_points) # + [0] * len(negative_contact_points)
        # contact_points = np.concatenate([positive_contact_points, negative_contact_points], axis=0)
        contact_points = positive_contact_points * 1.0
        
        return {
            'pcd': torch.from_numpy(pcd).float(),
            'contact_points': torch.from_numpy(contact_points).float(), # 2N, 4
            'labels': torch.tensor(labels).float()
        }
            
            
            
def main():
    
    np.random.seed(28)
    random.seed(28)
    torch.manual_seed(28)
    
    val_meshes = open('/scratch/dualarm/DG16M/test_meshes.txt', 'r').readlines()
    MESH_PATH = '/scratch/dualarm/DG16M/dg16m/meshes'
    GRASP_PATH = '/scratch/dualarm/DG16M/dg16m/grasps'
    
    args = {
        'random_rotation': False,
    }
    
    val_dataset = ContactPointClassificationDG16M(mesh_path=MESH_PATH, 
                                            grasp_path=GRASP_PATH,
                                            meshes_to_take=val_meshes,
                                            is_training=False,
                                            args=args)

    val_loader = DataLoader(val_dataset, batch_size=16)
    
    data = next(iter(val_loader))

    for k, v in data.items():
        print(k, v.shape)   
        
    # torch.save(data, '../../classification_batch.pt')

if __name__ == "__main__":
    main()