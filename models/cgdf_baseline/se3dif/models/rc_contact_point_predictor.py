import torch
import torch.nn as nn
import torch.nn.functional as F
from .pointnet import PointNetfeat


class RcContactPointPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.local_object_pointnet = PointNetfeat(num_points=512)
        self.grasp_pointnet = PointNetfeat(num_points=512)
        
        self.fc1 = nn.Sequential(
            nn.Linear(1024, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ELU(),
            nn.Linear(512, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ELU(),
        )
        
        self.fc2 = nn.Sequential(
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64, bias=False),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.Linear(64, 6),
        )
        
        self.finger_point_fc = nn.Sequential(
            nn.Linear(120, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ELU(),
            nn.Dropout(0.1),
            
            nn.Linear(256, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ELU()
        )
        
    def forward(self, local_pcd, grasp_pcd, finger_point_enc, B, N):
        # local_pcd: (bs, N, 512, 3)
        # grasp_pcd: (bs, N, 512, 3)
        # finger_point_enc: (bs, N, 2, 60)
        
        local_pcd = local_pcd.reshape(B*N, 512, 3).permute(0, 2, 1) # (bs*N, 3, 512)
        grasp_pcd = grasp_pcd.reshape(B*N, 512, 3).permute(0, 2, 1) # (bs*N, 3, 512)
        finger_point_enc = finger_point_enc.reshape(B*N, -1) # (bs*N, 120)
        finger_point_enc = self.finger_point_fc(finger_point_enc) # (bs*N, 512)
        
        local_feat, _ = self.local_object_pointnet(local_pcd) # B*N, 512
        grasp_feat, _ = self.grasp_pointnet(grasp_pcd) # B*N, 512
        
        x = torch.cat((local_feat + 1e-1 * finger_point_enc, 
                       grasp_feat + 1e-1 * finger_point_enc), dim=-1) # B*N, 1024
        x = self.fc1(x) # B*N, 512
        x = x + 0.1 * (local_feat + grasp_feat) + finger_point_enc
        
        x = self.fc2(x) # B*N, 6
        
        return x # B*N, 6
    
    
def main():
    pass

if __name__ == "__main__":
    main()  