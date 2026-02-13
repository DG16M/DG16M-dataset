import torch
import torch.nn as nn
import torch.nn.functional as F
from .nets.pointnet import PointNetfeat, PointNetSetAbstraction
# from nets.pointnet import PointNetfeat

# class DualGPDClassifier(nn.Module):
#     def __init__(self, input_channel=6):
#         super().__init__()
#         self.cls = DualPointNetCls(num_points=1024, 
#                                    input_chann=input_channel, 
#                                    k=2,
#                                    output_logits=True)
        
#     def forward(self, x):
#         # x: (batch_size, 6, 1024)
#         return self.cls(x) 


class DualGPDClassifier(nn.Module):
    
    def __init__(self):
        super().__init__()
        in_channel = 3
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(512, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.drop2 = nn.Dropout(0.1)
        # self.fc3 = nn.Linear(256, num_class)
        self.fc3 = nn.Sequential(
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ELU(),
            nn.Linear(256, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ELU(),
            nn.Linear(256, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.Linear(128, 1),
        )

    def forward(self, x1, x2):
        B, _, _ = x1.shape
        x1 = torch.cat([x1, x2], dim=-1)
        norm = None
        l1_xyz, l1_points = self.sa1(x1, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        # x = F.log_softmax(x, -1)
        x = F.sigmoid(x)
        return x

class DualGPDClassifier_OLD(nn.Module):
    def __init__(self, input_chann=4):
        super().__init__()
        self.pointnet = PointNetfeat(num_points=512 + 256, 
                                     input_chann=input_chann)
        # self.object_pointnet = PointNetfeat(num_points=512)
        # self.grasps_pointnet = PointNetfeat(num_points=256)
        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ELU(),
            
            nn.Dropout(0.2), 
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.Linear(128, 32),
            nn.ELU(),
            nn.Linear(32, 1)
        )
          
    def forward(self, x1, x2):
        # x1: shape points: (bs, 3, 512)
        # x2: grasp points: (bs, 3, 256)
        bs = x1.shape[0]
        ones_row = torch.ones((bs, 1, 512), device=x1.device, dtype=x1.dtype)
        zeros_row = torch.zeros((bs, 1, 256), device=x2.device, dtype=x2.dtype)
        x1 = torch.cat([x1, ones_row], dim=1)  # (bs, 4, 512)
        x2 = torch.cat([x2, zeros_row], dim=1) # (bs, 4, 256)
        
        x1 = torch.cat([x1, x2], dim=-1) # (bs, 4, 512+256)
        object_feat, _ = self.pointnet(x1) # bs, 1024
        
        return F.sigmoid(self.fc(object_feat)) # bs, 1
    
def num_params(model):
    trainable_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    non_trainable_params = sum([p.numel() for p in model.parameters() if not p.requires_grad])
    print("Trainable parameters: ", trainable_params)
    print("Non-trainable parameters: ", non_trainable_params)
    
def main():
    device = 'cuda'
    model = DualGPDClassifier().to(device)
    # B = 32
    # x = torch.randn(B, 6, 1024).to(device)
    
    # y = model(x)
    # print(y.shape)  # torch.Size([B, 2])
    num_params(model)
    
if __name__ == '__main__':
    main()