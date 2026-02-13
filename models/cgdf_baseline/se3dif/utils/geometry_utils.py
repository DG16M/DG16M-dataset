import torch

import numpy as np
import torch
import math
import json

import theseus as th
from theseus.geometry import SO3


class SO3_R3():
    def __init__(self, R=None, t=None):
        # Reproduce
        self.R = SO3(dtype=torch.float32)
        if R is not None:
            self.R.update(R)
        self.w = self.R.log_map()
        if t is not None:
            self.t = t

    def log_map(self):
        return torch.cat((self.t, self.w), -1)

    def exp_map(self, x):
        self.t = x[..., :3]
        self.w = x[..., 3:]
        self.R = SO3().exp_map(self.w)
        return self

    def to_matrix(self):
        H = torch.eye(4).unsqueeze(0).repeat(self.t.shape[0], 1, 1).to(self.t)
        H[:, :3, :3] = self.R.to_matrix()
        H[:, :3, -1] = self.t
        return H

    # The quaternion takes the [w x y z] convention
    def to_quaternion(self):
        return self.R.to_quaternion()

    def sample(self, batch=1):
        R = SO3().rand(batch)
        t = 2 * torch.randn(batch, 3)
        H = torch.eye(4).unsqueeze(0).repeat(batch, 1, 1).to(t)
        H[:, :3, :3] = R.to_matrix()
        H[:, :3, -1] = t
        return H
    
def filter_grasps_by_axis(grasps, conditions, tol=0.3):
    def dir_vector(cond):
        axis_map = {"x": np.array([1,0,0]),
                    "y": np.array([0,1,0]),
                    "z": np.array([0,0,1])}
        sign = 1 if cond[0] == "+" else -1
        axis = cond[1]
        return sign * axis_map[axis]

    target_dirs = [dir_vector(c) for c in conditions]
    
    mask = []
    for idx, g_pair in enumerate(grasps):
        y0 = g_pair[0][:3, 1]  # first grasp's y-axis
        y1 = g_pair[1][:3, 1]  # second grasp's y-axis
        
        cond0 = y0 @ target_dirs[0] > 1 - tol
        cond1 = y1 @ target_dirs[1] > 1 - tol
        if cond0 and cond1:
            mask.append(idx)
            continue

        cond0 = y1 @ target_dirs[0] > 1 - tol
        cond1 = y0 @ target_dirs[1] > 1 - tol
        
        if cond0 and cond1:
            mask.append(idx)
        
    mask = np.array(mask)
    
    distances = np.linalg.norm(grasps[mask][:, 0, :3, 3] - grasps[mask][:, 1, :3, 3], axis=-1)
    return np.argsort(distances)[::-1]
    return mask