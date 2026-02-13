import torch
import torch.nn as nn
import os 
import numpy as np
from tqdm import tqdm 
from icecream import ic
import copy
import wandb 
import trimesh

class CollisionGraspTrainer:
    def __init__(self,
                 args,
                 model,
                 train_loader,
                 val_loader,
                 device):
        self.args = args
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device    
        self.model = model.to(self.device)
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        self.loss_fxn = nn.BCELoss()
        
        self.gripper_pcd = trimesh.load(self.args['gripper_pcd_path'])
        self.gripper_pcd = torch.from_numpy(self.gripper_pcd.sample(256)).float()
        
    def batch_transform_points(self, points: torch.Tensor, transforms: torch.Tensor) -> torch.Tensor:
        B = transforms.shape[0]
        P = points.shape[0]

        # Convert to homogeneous coordinates: (1, P, 4)
        points_h = torch.cat([points, torch.ones(P, 1, device=points.device)], dim=-1)  # (P, 4)
        points_h = points_h.unsqueeze(0).expand(B, P, 4)  # (B, P, 4)

        # Apply batched transform: (B, P, 4) = (B, P, 4) x (B, 4, 4)^T
        points_h = torch.bmm(points_h, transforms.transpose(1, 2))  # (B, P, 4)

        return points_h[..., :3]  # (B, P, 3)
    
    def training_step(self, data):
        self.optimizer.zero_grad()
        
        pcds, grasps, labels = data.values() # pcd: (bs, 512, 3), grasps: (bs, 4, 4), labels: (bs,)
        grasps = self.batch_transform_points(self.gripper_pcd.clone(), grasps) # (bs, 256, 3)
        
        # torch.save({
        #     'pcd': pcds, 
        #     'grasp': grasps,
        #     'label': labels
        # }, './temp/collision_batch.pth')
        # exit()
        
        pcds, grasps = pcds.permute(0, 2, 1), grasps.permute(0, 2, 1) # (bs, 3, 512), (bs, 3, 256)
        pcds, grasps, labels = pcds.to(self.device), grasps.to(self.device), labels.to(self.device)
        
        outputs = self.model(pcds, grasps).flatten()
        loss = self.loss_fxn(outputs, labels)
        acc = ((outputs > 0.5).float() == labels).float().mean()
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item(), acc.item()
    
    def val_step(self, data):
        pcds, grasps, labels = data.values() # pcd: (bs, 512, 3), grasps: (bs, 4, 4), labels: (bs,)
        grasps = self.batch_transform_points(self.gripper_pcd.clone(), grasps) # (bs, 256, 3)
        
        pcds, grasps = pcds.permute(0, 2, 1), grasps.permute(0, 2, 1)
        pcds, grasps, labels = pcds.to(self.device), grasps.to(self.device), labels.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(pcds, grasps).flatten()
            loss = self.loss_fxn(outputs, labels)
            acc = ((outputs > 0.5).float() == labels).float().mean()
            
        return loss.item(), acc.item()
    
    def go_one_epoch(self, loader, step_fxn, text):
        loss, step, accuracy = 0, 0, 0
        for data in tqdm(loader, colour='cyan', desc=text):
            l, a = step_fxn(data)
            step +=1
            loss += l
            accuracy += a
            
            if step % 100 == 0:
                print(f"Step {step} | Loss: {loss/step} | Accuracy: {accuracy/step} | text: {text}")
                
        return loss/len(loader), accuracy/len(loader)
    
    def train(self, epochs):
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            
            self.model.train()
            train_loss, train_acc = self.go_one_epoch(self.train_loader, self.training_step, 'training')
            
            self.model.eval()
            val_loss, val_acc = self.go_one_epoch(self.val_loader, self.val_step, 'validation')
            
            # if self.args['use_wandb']:
            #     wandb.log({
            #         "train_loss": train_loss,
            #         "train_accuracy": train_acc,
            #         "val_loss": val_loss,
            #         "val_accuracy": val_acc,
            #         "epoch": epoch
            #     })
                
            print(f"Train Loss: {train_loss}, Train Accuracy: {train_acc}")
            print(f"Val Loss: {val_loss}, Val Accuracy: {val_acc}")
            
            if (epoch + 1) % 10 == 0:
                model_file_name = os.path.join(self.args['save_model_path'], f"model_epoch_{epoch+1}.pt")
                torch.save(self.model.state_dict(), model_file_name)