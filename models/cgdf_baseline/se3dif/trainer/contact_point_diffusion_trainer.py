import torch
import torch.nn as nn
import os 
import numpy as np
from tqdm import tqdm 
from icecream import ic
import copy
import wandb 

class ContactPointDiffusionTrainer:
    def __init__(self,
                 args,
                 model,
                 train_loader,
                 val_loader, 
                 diffusion,
                 device):
        self.args = args
        self.model = nn.DataParallel(model)
        self.train_loader = train_loader
        self.val_loader = val_loader    
        self.diffusion = diffusion
        self.device = device
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        self.loss_fxn = nn.MSELoss(reduction='mean')
        
        self.log_interval = 30
        if args['load_model_path'] is not None:
            ret = self.model.load_state_dict(torch.load(args['load_model_path']), strict=False)
            print(f"Model loaded from {args['load_model_path']}")
            print(ret)
            
        print(f"Using logger: {self.args['use_logger']}")
        if self.args['use_logger']:
            self.logger = wandb.init(
                project="contact_point_diffusion",
                config=self.args)
        else:
            self.logger = None
            
        
    def positional_encoding(self, xyz, num_freqs=10):
        freq_bands = 2 ** torch.arange(num_freqs, dtype=torch.float32, device=xyz.device) * np.pi  # shape (L,)
        xyz_proj = xyz[..., None] * freq_bands  # (B, N, 3, L)
        xyz_encoded = torch.cat([torch.sin(xyz_proj), torch.cos(xyz_proj)], dim=-1)
        xyz_encoded = xyz_encoded.view(xyz.shape[0], xyz.shape[1], -1)

        return xyz_encoded.reshape(-1, 2 * 3 * num_freqs)
    
    def training_step(self, data):
        self.optimizer.zero_grad()
        pcds, contact_points, labels = data.values() # (bs, 1024, 3), (bs, N, 4, 3), (bs, )
        B, N = pcds.shape[0], contact_points.shape[1]
        # cp_temp = contact_points.reshape(-1, 3) # (bs*4*N, 3)
        
        labels = labels.flatten()
        
        pcds, contact_points, labels = pcds.to(self.device), contact_points.to(self.device), labels.to(self.device)
        
        time = self.diffusion.sample_timesteps(B*N).to(self.device)
        # time = self.diffusion.sample_continuous_timesteps(B*N).to(self.device)
        contact_points_t, noise = self.diffusion.forward_images(contact_points.reshape(B*N, 4, 3), 
                                                        time)
        
        # contact_points_t, epsilon, std = self.diffusion.forward_process_score(
        #     x0=contact_points.reshape(B*N, 4, 3),
        #     t=time)
        
        contact_points_t = self.positional_encoding(contact_points_t) # (bs*4*N, 192)
        contact_points_t = contact_points_t.reshape(B, N, 4, -1)
        
        pred = self.model(pcds, contact_points_t, N, time) # (bs*N, 12)
        
        loss = self.loss_fxn(pred, noise.reshape(-1, 12))
        # loss = self.loss_fxn(pred_noise, noise.flatten())
        # score_target = epsilon / std
        
        # loss = self.loss_fxn(score_pred.reshape(-1, 12), score_target.reshape(-1, 12))
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def val_step(self, data):
        pcds, contact_points, labels = data.values() # (bs, 1024, 3), (bs, N, 4, 3), (bs, )
        B, N = pcds.shape[0], contact_points.shape[1]
        # cp_temp = contact_points.reshape(-1, 3) # (bs*4*N, 3)
        
        labels = labels.flatten()
        
        pcds, contact_points, labels = pcds.to(self.device), contact_points.to(self.device), labels.to(self.device)
        
        time = self.diffusion.sample_timesteps(B*N).to(self.device)
        # time = self.diffusion.sample_continuous_timesteps(B*N).to(self.device)
        contact_points_t, noise = self.diffusion.forward_images(contact_points.reshape(B*N, 4, 3), 
                                                        time)
        
        # contact_points_t, epsilon, std = self.diffusion.forward_process_score(
            # x0=contact_points.reshape(B*N, 4, 3),
            # t=time)
        
        contact_points_t = self.positional_encoding(contact_points_t) # (bs*4*N, 192)
        contact_points_t = contact_points_t.reshape(B, N, 4, -1)
        
        with torch.no_grad():
            
            pred = self.model(pcds, contact_points_t, N, time) # (bs*N, 12)
            loss = self.loss_fxn(pred, noise.reshape(-1, 12))
            # loss = self.loss_fxn(pred_noise, noise.flatten())
            # score_target = epsilon / std   
            
            # loss = self.loss_fxn(score_pred.reshape(-1, 12), score_target.reshape(-1, 12))
        
        return loss.item()
    
    def go_one_epoch(self, loader, step_fxn, text):
        loss, step = 0, 0
        
        for data in tqdm(loader, colour='cyan', desc=text):
            loss += step_fxn(data)
            step += 1
            
            if step % self.log_interval == 0 :
                print(f"Step: {step}: Loss: {loss/step} | text: {text}")
                if self.logger:
                    self.logger.log({
                        "step_loss": loss/step,
                    })

            if text == 'training' and step % 1000 == 0:
                model_file_name = os.path.join(self.args['save_model_path'], "current.pt")
                torch.save(self.model.state_dict(), model_file_name)
                
        return loss/len(loader)
    
    def train(self, num_epochs=100):
        for epoch in range(self.args['train_epochs']):
            self.model.train()
            train_loss = self.go_one_epoch(self.train_loader, self.training_step, text='training')
            self.model.eval()
            val_loss = self.go_one_epoch(self.val_loader, self.val_step, text='validation')
            
            print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss}")
            print(f"Epoch {epoch+1}/{num_epochs} | Val Loss: {val_loss}")
            print("---------------------------------------------------")
            
            if self.logger:
                self.logger.log({
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                })
                
            with open(self.args['exp_dir'] + '/log.txt', 'a') as f:
                f.write(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss}\n")
                f.write(f"Epoch {epoch+1}/{num_epochs} | Val Loss: {val_loss}\n")
                f.write("---------------------------------------------------\n")
                
            if self.args['save_model_path'] is not None:
                model_file_name = os.path.join(self.args['save_model_path'], f"{epoch+1}_{val_loss:.4f}.pt")
                torch.save(self.model.state_dict(), model_file_name)
            print(f"Model saved at {model_file_name}")