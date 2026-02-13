import torch
import torch.nn as nn
import torch.nn.functional as F
from .pointnet import PointNetfeat

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
    def __init__(self, input_channel=6):
        super().__init__()
        self.object_pointnet = PointNetfeat(num_points=1024)
        self.grasps_pointnet = PointNetfeat(num_points=512)
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
            nn.Linear(128, 1)
        )
          
    def forward(self, x1, x2):
        # x1: shape points: (bs, 3, 1024)
        # x2: grasp points: (bs, 3, 1024)
        object_feat, _ = self.object_pointnet(x1) # bs, 1024
        grasp_feat, _ = self.grasps_pointnet(x2) # bs, 1024
        common = object_feat + grasp_feat # bs, 2048
        # print('shape of fused', common.shape)
        # exit()
        return F.sigmoid(self.fc(common)) # bs, 1
    
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