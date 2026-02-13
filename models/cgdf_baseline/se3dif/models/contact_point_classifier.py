import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json 

from se3dif import models
from icecream import ic

def count_params(model):
    return sum([p.numel() for p in model.parameters() if p.requires_grad])


class ScoreNet(nn.Module):
    def __init__(self, x_dim, time_dim):
        super().__init__()
        self.x_dim = x_dim
        self.time_dim = time_dim
        
    def forward(self, x, t):
        pass


class CrossAttentionBlock(nn.Module):
    def __init__(self, query_dim, context_dim, num_heads=4, hidden_dim=None):
        super().__init__()
        self.query_dim = query_dim
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim or query_dim

        self.query_proj = nn.Linear(query_dim, self.hidden_dim)
        self.key_proj = nn.Linear(context_dim, self.hidden_dim)
        self.value_proj = nn.Linear(context_dim, self.hidden_dim)

        self.out_proj = nn.Linear(self.hidden_dim, query_dim)
        self.attn = nn.MultiheadAttention(embed_dim=self.hidden_dim,
                                          num_heads=num_heads,
                                          batch_first=True)

    def forward(self, queries, context):
        """
        queries: (B, 4, C_q)     # contact point features
        context: (B, N, C_k)     # PCD features (flattened planes or point features)
        returns: (B, 4, C_q)     # updated contact point features
        """

        q = self.query_proj(queries)   # (B, 4, D)
        k = self.key_proj(context)    # (B, N, D)
        v = self.value_proj(context)  # (B, N, D)

        # Apply attention
        out, _ = self.attn(q, k, v)    # (B, 4, D)

        # Residual + projection
        out = self.out_proj(out) + queries  # residual connection
        return out
    
class ContactPointClassifierDG16M(nn.Module):
    def __init__(self, 
                 args, 
                 load_pretrained_vision_encoder):
        super().__init__()
        self.args = args
        device = 'cuda'
        
        plane_type   = ['xz', 'xy', 'yz']
        grid_resolution = 32
        plane_resolution = 32
        latent_size = 132
        self.pcd_encoder = models.vision_encoder.VNNLocalPoolPointnet(c_dim = int(latent_size/3),
                                                                    grid_resolution=grid_resolution,
                                                                    plane_type=plane_type,
                                                                    unet=True,
                                                                    plane_resolution=plane_resolution, 
                                                                    device=device,
                                                                    unet_depth=4).to(device)
        
        if load_pretrained_vision_encoder is not None:
            weights_all = torch.load(load_pretrained_vision_encoder, map_location='cpu')                    
            ret = self.pcd_encoder.load_state_dict(weights_all, strict=False)
            print('Loaded vision encoder weights:', ret)
            
        else:
            print("Not loading pretrained vision encoder")
            
        self.pcd_encoder = self.pcd_encoder.to(device)
        self.contact_point_mlp = self.create_contact_mlp()
        
        self.fusion_mlp = self.create_contact_mlp(in_dim=128 * 4, out_dim=132 * 3)
        final_conv_in_dim = 132 * 3
        self.conv_block = nn.Sequential(
            nn.Conv2d(final_conv_in_dim, final_conv_in_dim//2, bias=False, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(final_conv_in_dim//2),
            nn.ELU(),
            
            nn.Conv2d(final_conv_in_dim//2, final_conv_in_dim//2, bias=False, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(final_conv_in_dim//2),
            nn.ELU(),
            
            nn.Conv2d(final_conv_in_dim//2, final_conv_in_dim//4, bias=False, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(final_conv_in_dim//4),
            nn.ELU(),
            
            nn.Conv2d(final_conv_in_dim//4, final_conv_in_dim//4, bias=False, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(final_conv_in_dim//4),
            nn.ELU()
        )
        self.average_pool = nn.AdaptiveAvgPool2d((1, 1))        
        
        self.noise_mlp = nn.Sequential(
            models.vision_encoder.layers.ResnetBlockFC(128*4, 512, activation=nn.ELU()),
            nn.ELU(),
            models.vision_encoder.layers.ResnetBlockFC(512, 256, activation=nn.ELU()),
            nn.ELU(),
            models.vision_encoder.layers.ResnetBlockFC(256, 128, activation=nn.ELU()),
            nn.ELU(),
            nn.Linear(128, 12)
        )
        
        self.classifier_mlp = nn.Sequential(
            models.vision_encoder.layers.ResnetBlockFC(128*4, 512, activation=nn.ELU()),
            nn.ELU(),
            models.vision_encoder.layers.ResnetBlockFC(512, 256, activation=nn.ELU()),
            nn.ELU(),
            models.vision_encoder.layers.ResnetBlockFC(256, 128, activation=nn.ELU()),
            nn.ELU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        
        self.projection_mlp = nn.Linear(132 * 3, 99)
        self.time_fc1 = nn.Sequential(
            nn.Linear(128, 128),
            nn.ELU(),
        )
        
        self.time_fc2 = nn.Sequential(
            nn.Linear(128, 132 * 3),
            nn.ELU()
        )
        
        self.time_fc3 = nn.Sequential(
            nn.Linear(128, 99),
            nn.ELU()
        )
        
        self.cross_attention = CrossAttentionBlock(query_dim=128, context_dim=99, num_heads=4)
    
        
    def create_contact_mlp(self, in_dim=60, out_dim=128):
        resnet_block1 = models.vision_encoder.layers.ResnetBlockFC(in_dim, in_dim, activation=nn.ELU())
        resnet_block2 = models.vision_encoder.layers.ResnetBlockFC(in_dim, out_dim//2, activation=nn.ELU())
        resnet_block3 = models.vision_encoder.layers.ResnetBlockFC(out_dim//2, out_dim, activation=nn.ELU())
        resnet_block4 = models.vision_encoder.layers.ResnetBlockFC(out_dim, out_dim, activation=nn.ELU())
        
        return nn.Sequential(
            resnet_block1,
            nn.BatchNorm1d(in_dim),
            nn.ELU(),
            nn.Dropout(0.1),
            
            resnet_block2,
            nn.BatchNorm1d(out_dim//2),
            nn.ELU(),
            nn.Dropout(0.1),
            
            resnet_block3,
            nn.BatchNorm1d(out_dim),
            nn.ELU(),
            nn.Dropout(0.1),
                        
            resnet_block4,
            nn.BatchNorm1d(out_dim),
            nn.ELU(),
        )
        
    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=t.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc
                
    def forward(self, pcd, contact_points, N, time):
        # pcd: (bs, 1024, 3), contact_points: (bs*4*N, 192)
        bs = pcd.shape[0]
        time = self.pos_encoding(time.reshape(bs*N, 1), channels=128) # (bs*N, 128)
        
        time = self.time_fc1(time).unsqueeze(1) # (bs*N, 1, 128)
        time1 = time.repeat(1, 4, 1).reshape(bs*N*4, -1) # (bs*N*4, 128) 
        
        contact_points = contact_points.reshape(bs*N*4, -1) # (bs*N*4, 60)
        contact_points = self.contact_point_mlp(contact_points) + time1 # (bs*4*N, 128)
        # contact_points = self.fusion_mlp(contact_points.reshape(bs*N, 128 * 4)) # bs*N, 132*3
        
        pcd_feature_planes, _ = self.pcd_encoder(pcd)
        pcd_feature_planes = torch.cat([pcd_feature_planes[plane] 
                                        for plane in ['xz', 'xy', 'yz']], dim=1) # (bs, 132*3, 32, 32)
        
        pcd_feature_planes = pcd_feature_planes.view(bs, -1, 32, 32)
        pcd_feature_planes = self.conv_block(pcd_feature_planes) # (bs, 99, 4, 4)
        pcd_feature_planes = pcd_feature_planes.reshape(-1, 1, 99, 16).repeat(1, N, 1, 1).reshape(bs*N, 99, 16)
        pcd_feature_planes = torch.permute(pcd_feature_planes, (0, 2, 1)) # (bs*N, 16, 99)
        
        contact_points = contact_points.reshape(bs*N, 4, -1) # (bs*N, 4, 128)
        
        x = self.cross_attention(contact_points, pcd_feature_planes) # (bs*N, 4, 128)
        
        x = x + time1.reshape(bs*N, 4, 128) # (bs*N, 4, 128)
        x = self.noise_mlp(x.reshape(bs*N, -1)) # (bs*N, 12)
        
        classifier_out = self.classifier_mlp(x.reshape(bs*N, -1)) # (bs*N, 1)
        
        return x, classifier_out
        
        # 1. contact_points: (bs*N, 132*3)
        # 2. pcd_feature_planes: (bs*N, 132*3, 32, 32)
        
        
    def forward2(self, pcd, contact_points, N, time):
        # pcd: (bs, 1024, 3), contact_points: (bs*4*N, 192)
        bs = pcd.shape[0]
        time = self.pos_encoding(time.reshape(bs*N, 1), channels=128) # (bs*N, 128)
        
        time = self.time_fc1(time).unsqueeze(1) # (bs*N, 1, 128)
        time1 = time.repeat(1, 4, 1).reshape(bs*N*4, -1) # (bs*N*4, 128) 
        
        contact_points = contact_points.reshape(bs*N*4, -1) # (bs*N*4, 60)
        contact_points = self.contact_point_mlp(contact_points) + time1 # (bs*4*N, 128)
        contact_points = self.fusion_mlp(contact_points.reshape(bs*N, 128 * 4)) # bs*N, 132*3
        
        pcd_feature_planes, _ = self.pcd_encoder(pcd)
        pcd_feature_planes = torch.cat([pcd_feature_planes[plane] 
                                        for plane in ['xz', 'xy', 'yz']], dim=1) # (bs, 132*3, 32, 32)
        
        pcd_feature_planes = pcd_feature_planes.view(bs, 1, -1, 32, 32)
        
        time2 = self.time_fc2(time.squeeze(1)) # (bs*N, 132*3)
        
        x = pcd_feature_planes.repeat(1, N, 1, 1, 1) + \
                    0.3 * contact_points.view(bs, N, -1, 1, 1) + \
                    0.5 * time2.view(bs, N, -1, 1, 1)
                    
        x = x.view(bs * N, -1, 32, 32)
        x = self.conv_block(x)
        x = self.average_pool(x)
        
        x = x.view(bs, N, -1) # (bs, N, 99)
        
        time3 = self.time_fc3(time.squeeze(1)).reshape(bs, N, -1) # (bs, N, 99)
        x = x + self.projection_mlp(contact_points.reshape(bs, N, -1)) + time3
        x = self.noise_mlp(x)
        return x
        
        
        
        
        
        
            
        
