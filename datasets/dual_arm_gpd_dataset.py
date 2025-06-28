import os
import torch
import h5py
from torch.utils.data import Dataset, DataLoader
import trimesh 
from tqdm import tqdm
import numpy as np
import random
from scipy.spatial.transform import Rotation as R


def transform_points(points, transform):
    assert points.shape[1] == 3, "Input points must have shape Nx3"
    assert transform.shape == (4, 4), "Transformation matrix must be 4x4"

    # Convert Nx3 to Nx4 (homogeneous coordinates)
    ones = np.ones((points.shape[0], 1))
    points_homogeneous = np.hstack([points, ones])  # Nx4

    # Apply transformation
    transformed_points_homogeneous = (transform @ points_homogeneous.T).T  # Nx4

    # Convert back to Nx3 (ignore homogeneous coordinate)
    return transformed_points_homogeneous[:, :3]

def transformation_matrix(i):
    angle = np.random.uniform(0, 0.6, 1)[0]
    transformation_matrix = np.zeros((4, 4))
    angles = np.random.uniform(-angle, angle, 3)  # Small angles for x, y, z

    Rx = np.array([[1, 0, 0, 0],
                [0, np.cos(angles[0]), -np.sin(angles[0]),0],
                [0, np.sin(angles[0]), np.cos(angles[0]), 0],
                [0, 0, 0, 1]])

    Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1]), 0],
                [0, 1, 0, 0],
                [-np.sin(angles[1]), 0, np.cos(angles[1]), 0],
                [0, 0, 0, 1]])

    Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0, 0],
                [np.sin(angles[2]), np.cos(angles[2]), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]])

    transformation_matrix = Rz @ Ry @ Rx
    return transformation_matrix
    

class DualGraspGpdDataset(Dataset):
    def __init__(self, mesh_path, grasp_path, meshes_to_take, args, is_training=True):
        super().__init__()
        self.mesh_path = mesh_path
        self.grasp_path = grasp_path
        self.args = args
        self.meshes_to_take = meshes_to_take
        self.mesh_files = [os.path.join(mesh_path, i) for i in self.meshes_to_take]
        
        self.obj_meshes = []
        self.grasps = []
        self.label = []
        self.gripper = trimesh.load(self.args['gripper_pcd_path'])
        self.is_training = is_training
        self.positive_grasps_num = 500
        self.negative_grasps_num = 500
        self.prune = np.random.randint(1, 1000, 200)
        self.augment_grasps = False
                
        for mesh_file in tqdm(self.mesh_files):
            mesh_file = mesh_file.strip()
            obj = trimesh.load(mesh_file)
            grasp_file = os.path.join(grasp_path, os.path.basename(mesh_file).replace('.obj', '.h5'))
            grasp_file = h5py.File(grasp_file, 'r')
            grasps = grasp_file['grasps/grasps'][()]
            scale = grasp_file['object/scale'][()]
            passed_indices = grasp_file['grasps/fc_passing_indices'][()]
            failed_indices = grasp_file['grasps/fc_failed_indices'][()] 
            # obj.apply_scale(scale)
            # obj.apply_translation(-obj.centroid)
            
            # take good grasps

            if len(passed_indices) == 0:
                # print(f"No good grasps for {mesh_file}")
                continue
            
            if len(passed_indices) < self.positive_grasps_num:
                self.grasps.extend(grasps[passed_indices])
                self.obj_meshes.extend([obj] * len(passed_indices))
                self.label.extend([1] * len(passed_indices))
            else:
                indices_to_take = np.random.choice(passed_indices, self.positive_grasps_num, replace=False)
                self.grasps.extend(grasps[indices_to_take])
                self.obj_meshes.extend([obj] * self.positive_grasps_num)
                self.label.extend([1] * self.positive_grasps_num)
            
            # take bad grasps
            if len(failed_indices) < self.negative_grasps_num:
                self.grasps.extend(grasps[failed_indices])
                self.obj_meshes.extend([obj] * len(failed_indices))
                self.label.extend([0] * len(failed_indices))
            else:
                indices_to_take = np.random.choice(failed_indices, self.negative_grasps_num, replace=False)
                self.grasps.extend(grasps[indices_to_take])
                self.obj_meshes.extend([obj] * self.negative_grasps_num)
                self.label.extend([0] * self.negative_grasps_num)
            
            
        self.gripper_left_pcd = np.asarray(self.gripper.sample(256))
        self.gripper_right_pcd = np.asarray(self.gripper.sample(256))
        
        self.n_samples = len(self.obj_meshes)
        print(f"Number of samples: {self.n_samples} | is_train: {self.is_training}")
        assert len(self.obj_meshes) == len(self.grasps) == len(self.label), "Length mismatch"
        
        # shuffle the lists
        indices = random.sample(range(self.n_samples), self.n_samples)
        self.obj_meshes = np.array(self.obj_meshes)[indices]
        self.grasps = np.array(self.grasps)[indices]
        self.label = np.array(self.label)[indices]
        
        
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, index):
        random_R = R.random().as_matrix()
        # random_R = np.eye(3)
        random_H = np.eye(4)
        random_H[:3, :3] = random_R

        obj = self.obj_meshes[index]
        obj.apply_transform(random_H)

        grasp = self.grasps[index]
        grasp[0, ...] = random_H @ grasp[0, ...]
        grasp[1, ...] = random_H @ grasp[1, ...]

        pcd = np.asarray(obj.sample(1024))
        label = self.label[index]

        return torch.from_numpy(pcd).float(), torch.from_numpy(grasp).float(), torch.tensor(label).float()
    
    # def __getitem__(self, index):
    #     obj = self.obj_meshes[index]
    #     grasp = self.grasps[index]
        
    #     grasp_left = transform_points(self.gripper_left_pcd.copy(), grasp[0])
    #     grasp_right = transform_points(self.gripper_right_pcd.copy(), grasp[1]) 
        
    #     grasp = torch.cat([torch.from_numpy(grasp_left), 
    #                        torch.from_numpy(grasp_right)], dim=0) # 1024
        
    #     pcd = np.asarray(obj.sample(1024)) # 1024
    #     label = self.label[index] # 1
        
    #     return torch.from_numpy(pcd).float(), grasp.float(), torch.tensor(label).float()
    
    
def main():
    
    train_meshes = open('/home/mtron_lab/mahesh/meshes_split/train_2_da2.txt', 'r').readlines()
    dual_grasps_dataset = DualGraspGpdDataset(mesh_path='/scratch/dualarm/DA2_opt_dataset/scaled_meshes', 
                                            grasp_path="/home/mtron_lab/mahesh/17thfeb/1a3efcaaf8db9957a010c31b9816f48b.h5",
                                            meshes_to_take=train_meshes)
    
    train_loader = DataLoader(dual_grasps_dataset, batch_size=32, shuffle=True)
    print(train_loader)
    
    pcd, grasp, label = next(iter(train_loader))
    print(pcd.shape, grasp.shape, label.shape)
    torch.save(pcd, './temp_data_point/pcd.pt')
    torch.save(grasp, './temp_data_point/grasp.pt')
    torch.save(label, './temp_data_point/label.pt')
    
if __name__ == "__main__":
    main()
