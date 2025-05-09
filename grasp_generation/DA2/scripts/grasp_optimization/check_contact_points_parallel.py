import numpy as np
import os
from tqdm import tqdm
import h5py
import random
import trimesh
from grasp_optimization.force_closure_optmization import fc_optimization
import concurrent.futures

def set_seed(seed=28):
    random.seed(seed)
    np.random.seed(seed)
    
LOSS_THRESHOLD = 1e-5
NUM_WORKERS = 12

def select_good_and_bad_grasps(losses, k=2000, initial_threshold=0.1):
    """
    Selects 2K good grasps (randomly sampled from those with loss < threshold)
    and 2K bad grasps (highest loss values). If there aren't enough good or bad 
    grasps, thresholds are adjusted dynamically.
    
    Parameters:
        losses (list or np.array): List of loss values for different grasps.
        k (int): Number of grasps to select in each category (default is 2000).
        initial_threshold (float): Initial threshold for good grasps (default is 0.1).
    
    Returns:
        dict: Dictionary with indices of selected good and bad grasps.
    """
    losses = np.array(losses)
    sorted_indices = np.argsort(losses)  # Sort indices by loss values (ascending)

    # Select bad grasps (2K highest loss values)
    worst_indices = sorted_indices[-k:].tolist() if len(sorted_indices) >= k else sorted_indices.tolist()

    # Select good grasps (based on threshold)
    good_indices = np.where(losses <= initial_threshold)[0]

    # Adjust threshold dynamically if not enough good grasps
    while len(good_indices) < k:
        initial_threshold += 0.05  # Increase threshold
        good_indices = np.where(losses <= initial_threshold)[0]
        if initial_threshold >= np.max(losses):  # Stop if max threshold is reached
            break

    # Randomly sample 2K good grasps (or as many as available)
    good_indices = np.random.choice(good_indices, size=min(k, len(good_indices)), replace=False).tolist()

    # Handle case where there aren’t enough bad grasps
    if len(worst_indices) < k:
        print(f"Warning: Only {len(worst_indices)} bad grasps available instead of {k}.")

    return {
        "good_grasps": good_indices,
        "bad_grasps": worst_indices
    }

def run_fc_optimization_2(mesh, contact_points, object_mass=6, friction_coeff=0.4):
    force_closure_passing_indices = []
    loss_values = []
    contact_forces = []
    frames = []
    
    contact_normals = np.zeros((len(contact_points), 4, 3))
    face1 = mesh.nearest.on_surface(contact_points[:, 0, :])[-1]
    face2 = mesh.nearest.on_surface(contact_points[:, 1, :])[-1]
    face3 = mesh.nearest.on_surface(contact_points[:, 2, :])[-1]
    face4 = mesh.nearest.on_surface(contact_points[:, 3, :])[-1]
    
    contact_normals[:, 0, :] = mesh.face_normals[face1]
    contact_normals[:, 1, :] = mesh.face_normals[face2]
    contact_normals[:, 2, :] = mesh.face_normals[face3]
    contact_normals[:, 3, :] = mesh.face_normals[face4]
    
    print("Contact normals found, starting optimization")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:   
        results = list(executor.map(fc_optimization, 
                                    contact_points, 
                                    contact_normals, 
                                    [10 * object_mass] * len(contact_points), 
                                    [friction_coeff] * len(contact_points), 
                                    [False] * len(contact_points)))

    for i in tqdm(range(len(results))):
        f1, f2, f3, f4, loss, frame = results[i]
        if loss < LOSS_THRESHOLD and f1 is not None and f2 is not None and f3 is not None and f4 is not None:
            force_closure_passing_indices.append(i)
            
        loss_values.append(loss)
        contact_forces.append([f1, f2, f3, f4])
        frames.append(frame)

    # selected_indices = select_grasps_by_loss(loss_values)
    print(f'Total grasps passed: {len(force_closure_passing_indices)} out of {len(contact_points)}')
    
    loss_values = np.array(loss_values)
    contact_forces = np.array(contact_forces)
    frames = np.array(frames)
    
    return force_closure_passing_indices, loss_values, contact_forces, frames
   

def run_fc_optimization(mesh, contact_points, object_mass=6, friction_coeff=0.4, orientation = 0):
    force_closure_passing_indices = []
    loss_values = []
    contact_forces = []
    frames = []
    
    contact_normals = np.zeros((len(contact_points), 4, 3))
    face1 = mesh.nearest.on_surface(contact_points[:, 0, :])[-1]
    face2 = mesh.nearest.on_surface(contact_points[:, 1, :])[-1]
    face3 = mesh.nearest.on_surface(contact_points[:, 2, :])[-1]
    face4 = mesh.nearest.on_surface(contact_points[:, 3, :])[-1]
    
    contact_normals[:, 0, :] = mesh.face_normals[face1]
    contact_normals[:, 1, :] = mesh.face_normals[face2]
    contact_normals[:, 2, :] = mesh.face_normals[face3]
    contact_normals[:, 3, :] = mesh.face_normals[face4]
    
    print("Contact normals found, starting optimization")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:   
        results = list(executor.map(fc_optimization, 
                                    contact_points, 
                                    contact_normals, 
                                    [10 * object_mass] * len(contact_points), 
                                    [friction_coeff] * len(contact_points), 
                                    [False] * len(contact_points),
                                    [orientation]*len(contact_points)))

    for i in tqdm(range(len(results))):
        f1, f2, f3, f4, loss, frame = results[i]
        if f1 is not None and f2 is not None and f3 is not None and f4 is not None:
            if loss < LOSS_THRESHOLD:
                force_closure_passing_indices.append(i)
            
        loss_values.append(loss)
        contact_forces.append([f1, f2, f3, f4])
        frames.append(frame)

    print(f'Total grasps passed: {len(force_closure_passing_indices)} out of {len(contact_points)}')
    
    loss_values = np.array(loss_values)
    contact_forces = np.array(contact_forces)
    frames = np.array(frames)
    
    return force_closure_passing_indices, loss_values, contact_forces, frames
    
def main():
    LOSS_THRESHOLD = 1e-5
    NUM_WORKERS = 10
    
    obj = '1e2b3f2047c62de9594de057c402974e.obj'
    save_path = f'../../results/{obj}'
    os.makedirs(save_path, exist_ok=True)
    
    obj_path = f'../../selected_meshes/{obj}'
    # grasp_path = f'.//grasps/{obj}.h5'
    grasp_path = f"../../generated_grasps/grasps_8feb/{obj.split('.')[0]}.h5"
    
    # for grasp_file in grasp_files:
    #     if obj.split('.')[0] in grasp_file:
    #         grasp_path = f'./DA2/grasps/{grasp_file}'
    #         break
    
    friction_coeff = 0.4
    object_mass = 6
    
    mesh = trimesh.load(obj_path)
    
    with h5py.File(grasp_path, 'r') as f:
        contact_points = f['grasps/grasp_points'][()]
        scale = f['object/scale'][()]
        
    mesh.apply_scale(scale)
    mesh.apply_translation(-mesh.centroid)    
    
    # contact_points = contact_points[:10000]
        
    
    force_closure_passing_indices = []
    loss_values = []
    contact_forces = []
    frames = []
    
    
    contact_normals = np.zeros((len(contact_points), 4, 3))
    # contact_points: (N, 4, 3)
    
    face1 = mesh.nearest.on_surface(contact_points[:, 0, :])[-1]
    face2 = mesh.nearest.on_surface(contact_points[:, 1, :])[-1]
    face3 = mesh.nearest.on_surface(contact_points[:, 2, :])[-1]
    face4 = mesh.nearest.on_surface(contact_points[:, 3, :])[-1]
    
    contact_normals[:, 0, :] = mesh.face_normals[face1]
    contact_normals[:, 1, :] = mesh.face_normals[face2]
    contact_normals[:, 2, :] = mesh.face_normals[face3]
    contact_normals[:, 3, :] = mesh.face_normals[face4]
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:   
        results = list(executor.map(fc_optimization, 
                                    contact_points, 
                                    contact_normals, 
                                    [10 * object_mass] * len(contact_points), 
                                    [friction_coeff] * len(contact_points), 
                                    [False] * len(contact_points)))

    for i in tqdm(range(len(results))):
        f1, f2, f3, f4, loss, frame = results[i]
        if loss < LOSS_THRESHOLD and f1 is not None and f2 is not None and f3 is not None and f4 is not None:
            force_closure_passing_indices.append(i)
            
        loss_values.append(loss)
        contact_forces.append([f1, f2, f3, f4])
        frames.append(frame)
        
    print(f'Total grasps passed: {len(force_closure_passing_indices)} out of {len(contact_points)}')
    
    print(force_closure_passing_indices)
    
    # force_closure_passing_indices = np.array(force_closure_passing_indices)
    # loss_values = np.array(loss_values)
    # contact_forces = np.array(contact_forces)
    
    # print(f'Total grasps passed: {len(force_closure_passing_indices)} out of {len(contact_points)}')
    
    # np.save(f'{save_path}/force_closure_passing_indices.npy', force_closure_passing_indices)
    # np.save(f'{save_path}/loss_values.npy', loss_values)
    # np.save(f'{save_path}/contact_forces.npy', contact_forces)
    # np.save(f'{save_path}/contact_frames.npy', frames)
    
if __name__ == '__main__':
    set_seed()
    main()