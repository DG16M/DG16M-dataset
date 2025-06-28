import trimesh
import torch
import torch.nn as nn
import wandb
from tqdm import tqdm 
import os


class PointnetGPDTrainer:
    def __init__(self, 
                 model,
                 train_loader, 
                 val_loader, 
                 device,
                 args,
                 log_interval=100,
                 val_interval=1000):
        
        self.device = device
        self.model = model.to(device)
        self.args = args        
        
        if self.args['load_model_path'] is not None:
            self.model.load_state_dict(torch.load(self.args['load_model_path']))
            print(f"Model loaded from {self.args['load_model_path']}")
        
        
        self.train_loader = train_loader  # Convert to iterator
        self.val_loader = val_loader
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), 1e-4)
        self.loss_fxn = nn.BCELoss()
        self.gripper_pcd = trimesh.load(self.args['gripper_pcd_path'])
        self.gripper_pcd = torch.from_numpy(self.gripper_pcd.sample(256)) 
        self.log_interval = log_interval 
        self.val_interval = val_interval
        if self.args['use_wandb']:
            wandb.init(project="DualGraspClassifier")
            self.logger = wandb
        else:
            self.logger = None
    
        self.model = nn.DataParallel(self.model)
        
    def batch_transform_points(self, points, transforms):
        P = points.shape[0] 
        points_h = torch.cat([points, torch.ones(P, 1, device=points.device)], dim=-1).float()  # (P, 4)
        transformed_h = torch.einsum("bnij,pj->bnpi", transforms, points_h)
        return transformed_h[..., :3]
        
    def training_step(self, data):
        self.optimizer.zero_grad()
        pcds, grasps, labels = data # (bs, 1024, 3), (bs, 2, 4, 4), (bs, )
        grasps = self.batch_transform_points(self.gripper_pcd.clone(), grasps) # (bs, 2, 256, 3)
        grasps = grasps.reshape(grasps.shape[0], -1, 3) # (bs, 512, 3)
        
        pcds, grasps = pcds.permute(0, 2, 1), grasps.permute(0, 2, 1)
        pcds, grasps, labels = pcds.to(self.device), grasps.to(self.device), labels.to(self.device)
        
        outputs = self.model(pcds, grasps).flatten()
        loss = self.loss_fxn(outputs, labels)
        acc = ((outputs > 0.5) == labels).sum().item() / len(labels)
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item(), acc     
    
    def val_step(self, data):
        pcds, grasps, labels = data # (bs, 1024, 3), (bs, 2, 4, 4), (bs, )
        grasps = self.batch_transform_points(self.gripper_pcd.clone(), grasps) # (bs, 2, 256, 3)
        grasps = grasps.reshape(grasps.shape[0], -1, 3) # (bs, 512, 3)
        
        pcds, grasps = pcds.permute(0, 2, 1), grasps.permute(0, 2, 1)
        pcds, grasps, labels = pcds.to(self.device), grasps.to(self.device), labels.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(pcds, grasps).flatten()
            loss = self.loss_fxn(outputs, labels)
            acc = ((outputs > 0.5) == labels).sum().item() / len(labels)
        
        # self.model.train()
        return loss.item(), acc
    
    def go_one_epoch(self, loader, step_func, text):
        loss, step, accuracy = 0, 0, 0
        for data in tqdm(loader, colour='cyan', desc=text):
            l, a = step_func(data)
            step +=1
            loss += l
            accuracy += a
            
            if step % 100 == 0:
                print(f"Step {step} | Loss: {loss/step} | Accuracy: {accuracy/step} | text: {text}")
                
        return loss/len(loader), accuracy/len(loader)
    
    def train(self, num_epochs=10):
        for epoch in range(num_epochs):
            self.model.train()
            train_loss, train_acc = self.go_one_epoch(self.train_loader, self.training_step, text='training')
            self.model.eval()
            val_loss, val_acc = self.go_one_epoch(self.val_loader, self.val_step, text='validation')
            
            # self.model.train()
            print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss} | Train Accuracy: {train_acc}")
            
            # self.model.eval()
            print(f"Epoch {epoch+1}/{num_epochs} | Val Loss: {val_loss} | Val Accuracy: {val_acc}")
            print("---------------------------------------------------")
            
            if self.logger:
                self.logger.log({
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc
                })
                
            with open(self.args['exp_dir'] + '/log.txt', 'a') as f:
                f.write(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss} | Train Accuracy: {train_acc}\n")
                f.write(f"Epoch {epoch+1}/{num_epochs} | Val Loss: {val_loss} | Val Accuracy: {val_acc}\n")
                f.write("---------------------------------------------------\n")
                
            if self.args['save_model_path'] is not None:
                model_file_name = os.path.join(self.args['save_model_path'], f"{epoch+1}_{val_acc:.4f}.pt")
                torch.save(self.model.state_dict(), model_file_name)
                print(f"Model saved at {model_file_name}")