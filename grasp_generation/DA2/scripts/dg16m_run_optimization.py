import os
import numpy as np
import sys
from dexnet.grasping import GraspableObject3D, RobotGripper
from autolab_core import YamlConfig
from meshpy import ObjFile
import h5py
from dexnet.api import DexNet
import yaml
from loguru import logger
import time 
from tqdm import tqdm
import pickle
from grasp_optimization.check_contact_points_parallel import run_fc_optimization
import random
import trimesh
from itertools import combinations


def set_seed(seed=2828):  
    random.seed(seed)
    np.random.seed(seed)
    

def f(OBJ_FILENAME, GRASP_FILENAME, SAVE_PATH, return_grasps=False, target_num_grasps=500):
    config_filename = "../api_config.yaml"
    
    if not os.path.isabs(config_filename):
        config_filename = os.path.join(os.getcwd(), config_filename)

    with open(config_filename, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        
    gripper_name = "robotiq_85"
    gripper = RobotGripper.load(gripper_name, "./grippers")

    # def mesh_antipodal_grasp_sampler():
    #     of = ObjFile(OBJ_FILENAME)
    #     mesh = of.read()

    #     obj = GraspableObject3D(None, mesh)
    #     logger.info("Starting grasp sampling")
    #     scale, grasps = DexNet._single_obj_grasps(None, obj, gripper, config, stable_pose_id=None, target_num_grasps=target_num_grasps)
    #     logger.info("Computed {} grasps".format(len(grasps)))

    #     return scale, grasps, gripper
    
    def get_existing_grasps(filename):
        grasp_file = h5py.File(filename, 'r')
        grasps = grasp_file['grasps/grasps'][()].reshape(-1, 4, 4)
        contact_points = grasp_file['grasps/contact_points'][()].reshape(-1, 2, 3)
        _, unique_indices = np.unique(grasps, axis=0, return_index=True)
        grasps = grasps[unique_indices]
        contact_points = contact_points[unique_indices]
        
        return 1.0, grasps, contact_points
        
   
    scale, grasp_transforms, contact_points = get_existing_grasps(GRASP_FILENAME)
    # create combinations of each grasp and contact_points
    print(grasp_transforms.shape, contact_points.shape)
    grasp_transforms = np.array([c for c in combinations(grasp_transforms, 2)])
    contact_points = np.array([c for c in combinations(contact_points, 2)]).reshape(-1, 4, 3)
    # remove grasps which are very close to each other
    distances = np.linalg.norm(grasp_transforms[:, 0, :3, 3] - grasp_transforms[:, 1, :3, 3], axis=1)
    indices_to_take = np.where(distances > 1e-1)[0]
    print(f"Removing {len(grasp_transforms) - len(indices_to_take)} grasps as they are close to each other.")
    grasp_transforms = grasp_transforms[indices_to_take]
    contact_points = contact_points[indices_to_take]
    print(grasp_transforms.shape, contact_points.shape)
    # exit()
    
    mesh = trimesh.load(OBJ_FILENAME)
    mesh.apply_scale(scale)
    mesh.apply_translation(-mesh.centroid)
    
    fc_passing_indices, loss_values, contact_forces, frames = run_fc_optimization(mesh, contact_points)
    fc_failed_indices = np.array([i for i in range(len(contact_points)) if i not in fc_passing_indices])
    
    if len(fc_passing_indices) > 2000:
        # fc_passing_indices = fc_passing_indices[np.where(loss_values > 2)[0]]
        fc_passing_indices = np.random.choice(fc_passing_indices, 2000, replace=False)
        
    fc_passing_grasps = grasp_transforms[fc_passing_indices]
    fc_passing_contact_points = contact_points[fc_passing_indices]
    fc_passing_contact_forces = contact_forces[fc_passing_indices]
    fc_passing_losses = loss_values[fc_passing_indices]
        
    
    if len(fc_failed_indices) > 2000:
        # fc_failed_indices = np.argsort(loss_values)[-min(10000, len(fc_failed_indices)):]]
        # fc_failed_indices = np.argsort(loss_values)[-int(len(loss_values)/2):]
        fc_failed_indices = np.where(np.array(loss_values) > 0.5)[0]
        probabilities = np.exp(-loss_values[fc_failed_indices])
        probabilities /= np.sum(probabilities)
        if len(fc_failed_indices) > 2000:
            fc_failed_indices = np.random.choice(fc_failed_indices, 2000, replace=False, p=probabilities)
            
    fc_failed_grasps = grasp_transforms[fc_failed_indices]
    fc_failed_contact_points = contact_points[fc_failed_indices]
    fc_failed_contact_forces = contact_forces[fc_failed_indices]
    fc_failed_losses = loss_values[fc_failed_indices]
    
    
    filename = os.path.join(SAVE_PATH, OBJ_FILENAME.split('.obj')[0].split('/')[-1] + '.h5')
    
    logger.info("Saving grasps to file: {}".format(filename))
    data = h5py.File(filename, 'w')
    temp1 = data.create_group("grasps")
    temp1['grasps'] = np.concatenate((fc_passing_grasps, fc_failed_grasps), axis=0) # 4000, 2, 4, 4
    temp1['contact_points'] = np.concatenate((fc_passing_contact_points, fc_failed_contact_points), axis=0) # 4000, 4, 3
    temp1['contact_forces'] = np.concatenate((fc_passing_contact_forces, fc_failed_contact_forces), axis=0) # 4000, 4, 3
    temp1['loss_values'] = np.concatenate((fc_passing_losses, fc_failed_losses), axis=0) # 4000
    temp1['fc_passing_indices'] = np.array([i for i in range(len(fc_passing_indices))]) # 0-1999: passing
    temp1['fc_failed_indices'] = np.array([i for i in range(len(fc_passing_indices), 
                                                            len(fc_passing_indices) + len(fc_failed_indices))]) # 2000-3999: failed 
    
    temp2 = data.create_group("object")
    temp2["file"] = OBJ_FILENAME.split('/')[-1]
    temp2["scale"] = scale
    
    return len(fc_passing_indices), len(fc_failed_indices)
    
    
def main():
    set_seed()
    OBJ_PATH = '/scratch/dualarm/DA2_opt_dataset/scaled_meshes'
    SAVE_PATH = '/scratch/dualarm/DG16M/april_10_remaining'
    GRASP_PATH = '/scratch/dualarm/DG16M/better_20thfeb'
    os.makedirs(SAVE_PATH, exist_ok=True)
    
    try:
        done_objects = open(os.path.join(SAVE_PATH, 'time_taken.txt')).read().split('\n')
        done_objects = [m.split(':')[0] for m in done_objects]  
    except:
        done_objects = None
        
    selected_objects = open('/home/dualarm/dummyhome/md/March2025/DG16M-dataset/notebooks/failed_grasps.txt').read().split()
    objects = os.listdir(OBJ_PATH)
    
    for object in objects:
        if object not in selected_objects:
            print('Not supposed to use this. Skipping !!!', object)
            continue
        if done_objects is not None:
            if object in done_objects:
                print("Already done. Skipping !!!", object)
                continue
        start_time = time.time()
        object_path = os.path.join(OBJ_PATH, object)
        grasp_path = os.path.join(GRASP_PATH, object.replace('.obj', '.h5'))
        print(f"Processing: {object_path} | {grasp_path}")
        try:
            num_passing, num_failed = f(object_path, grasp_path, SAVE_PATH, target_num_grasps=500)
            end_time = time.time()
            time_taken = end_time - start_time
            current_time = time.strftime('[%Y-%m-%d] [%H:%M:%S]', time.localtime())
            with open(os.path.join(SAVE_PATH, 'time_taken.txt'), 'a') as file:
                file.write(f"{object}: {time_taken} | passing: {num_passing} | failed: {num_failed} | time: {current_time}\n")
                    
        except Exception as e:
            with open(os.path.join(SAVE_PATH, 'time_taken.txt'), 'a') as file:
                file.write(f"{object}: Failed\n")
                print(f"Failed: {object} | {e}")
            
        file.close()
        
            
    
    
def main2():
    set_seed()
    
    OBJ_PATH = '/scratch/dualarm/DA2_opt_dataset/scaled_meshes'
    # SAVE_PATH = '/scratch/dualarm/DA2_opt_dataset/our_grasps/grasps_split4/'
    SAVE_PATH = '/scratch/dualarm/DG16M/split4'
    os.makedirs(SAVE_PATH, exist_ok=True)
    
    # selected_meshes = open(os.path.join(SAVE_PATH, 'time_taken.txt')).read().split('\n')
    # selected_meshes = [m.split(':')[0] for m in selected_meshes]
    try:
        done_objects = open(os.path.join(SAVE_PATH, 'time_taken.txt')).read().split('\n')
        done_objects = [m.split(':')[0] for m in done_objects]
    except:
        done_objects = None
        
        
    selected_meshes = open('/scratch/dualarm/DG16M/meshes_split/split_4.txt').read().split('\n')
    objects = os.listdir(OBJ_PATH)

    for object in objects:
        if object not in selected_meshes:
            print("Not supposed to use this. Skipping !!!", object)
            continue
        
        if done_objects is not None:
            if object in done_objects:
                print("Already done. Skipping !!!", object)
                continue
        
        start_time = time.time()
        object_path = os.path.join(OBJ_PATH, object)
        print(f"Processing: {object_path}")
        try:
            num_passing, num_failed = f(object_path, SAVE_PATH, target_num_grasps=500)
            end_time = time.time()
            time_taken = end_time - start_time
            current_time = time.strftime('[%Y-%m-%d] [%H:%M:%S]', time.localtime())
            with open(os.path.join(SAVE_PATH, 'time_taken.txt'), 'a') as file:
                file.write(f"{object}: {time_taken} | passing: {num_passing} | failed: {num_failed} | time: {current_time}\n")
                
        except Exception as e:
            with open(os.path.join(SAVE_PATH, 'time_taken.txt'), 'a') as file:
                file.write(f"{object}: Failed\n")
                print(f"Failed: {object} | {e}")
            
        file.close()
        
if __name__ == "__main__":
    main()
