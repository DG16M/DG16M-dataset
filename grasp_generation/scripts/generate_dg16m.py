import os
import numpy as np
from dexnet.grasping import GraspableObject3D, RobotGripper
from meshpy import ObjFile
import h5py
from dexnet.api import DexNet
import yaml
from loguru import logger
import time 
from grasp_optimization.check_contact_points_parallel import run_fc_optimization
import random
import trimesh
import argparse

def set_seed(seed=2828):  
    random.seed(seed)
    np.random.seed(seed)
    

def f(OBJ_FILENAME, SAVE_PATH, return_grasps=False, target_num_grasps=500, num_workers=8):
    config_filename = "../api_config.yaml"
    
    if not os.path.isabs(config_filename):
        config_filename = os.path.join(os.getcwd(), config_filename)

    with open(config_filename, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        
    gripper_name = "robotiq_85"
    gripper = RobotGripper.load(gripper_name, "./grippers")

    def mesh_antipodal_grasp_sampler():
        of = ObjFile(OBJ_FILENAME)
        mesh = of.read()

        obj = GraspableObject3D(None, mesh)
        logger.info("Starting grasp sampling")
        scale, grasps = DexNet._single_obj_grasps(None, obj, gripper, config, stable_pose_id=None, target_num_grasps=target_num_grasps, num_workers=8)
        logger.info("Computed {} grasps".format(len(grasps)))

        return scale, grasps, gripper


    scale, g, gripper = mesh_antipodal_grasp_sampler()
    
    g = np.array(g)
    contact_points = np.array([(g[i][0].grasp_point1, g[i][0].grasp_point2, 
                              g[i][1].grasp_point1, g[i][1].grasp_point2) for i in range(len(g))])
    grasp_transforms = np.array([((g[i][0].gripper_pose(gripper) * gripper.T_mesh_gripper.inverse()).matrix, 
                                  (g[i][1].gripper_pose(gripper) * gripper.T_mesh_gripper.inverse()).matrix) for i in range(len(g))])
    
    if return_grasps:
        return grasp_transforms, contact_points
    
    mesh = trimesh.load(OBJ_FILENAME)
    mesh.apply_scale(scale)
    mesh.apply_translation(-mesh.centroid)
    
    fc_passing_indices, loss_values, contact_forces, frames = run_fc_optimization(mesh, contact_points, num_workers)
    fc_failed_indices = [i for i in range(len(contact_points)) if i not in fc_passing_indices]
    
    if len(fc_passing_indices) > 2000:
        fc_passing_indices = np.random.choice(fc_passing_indices, 2000, replace=False)
        
    fc_passing_grasps = grasp_transforms[fc_passing_indices]
    fc_passing_contact_points = contact_points[fc_passing_indices]
    fc_passing_contact_forces = contact_forces[fc_passing_indices]
    fc_passing_losses = loss_values[fc_passing_indices]
        
    
    if len(fc_failed_indices) > 2000:
        # fc_failed_indices = np.argsort(loss_values)[-min(10000, len(fc_failed_indices)):]]
        # fc_failed_indices = np.argsort(loss_values)[-int(len(loss_values)/2):]
        fc_failed_indices = np.where(np.array(loss_values) > 0.5)[0]
        if len(fc_failed_indices) > 2000:
            fc_failed_indices = np.random.choice(fc_failed_indices, 2000, replace=False)

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
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--meshes_path', type=str, required=True, help='Path to the meshes folder')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save the grasps')
    parser.add_argument('--selected_meshes_txt', type=str, required=False, help='Path to the selected meshes txt file')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for parallel processing')
    
    args = parser.parse_args()
    
    OBJ_PATH = args.meshes_path
    SAVE_PATH = args.save_path
    os.makedirs(SAVE_PATH, exist_ok=True)
    
    try:
        done_objects = open(os.path.join(SAVE_PATH, 'time_taken.txt')).read().split('\n')
        done_objects = [m.split(':')[0] for m in done_objects]
    except:
        done_objects = None
        
    done_objects = None
        
    selected_meshes = open(args.selected_meshes_txt).read().split('\n') if args.selected_meshes_txt else None
    objects = os.listdir(OBJ_PATH)

    for object in objects:
        if selected_meshes is not None:
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
        num_passing, num_failed = f(object_path, SAVE_PATH, target_num_grasps=500, num_workers=args.num_workers)
        end_time = time.time()
        time_taken = end_time - start_time
        current_time = time.strftime('[%Y-%m-%d] [%H:%M:%S]', time.localtime())
        with open(os.path.join(SAVE_PATH, 'time_taken.txt'), 'a') as file:
            file.write(f"{object}: {time_taken} | passing: {num_passing} | failed: {num_failed} | time: {current_time}\n")
        
if __name__ == "__main__":
    main()
