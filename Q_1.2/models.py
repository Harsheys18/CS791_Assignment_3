import torch.nn as nn
import torch
import torch.nn.functional as F
from scheduler import NoiseSchedulerDDPM
import os
import torch.optim as optim
from tqdm import tqdm

class Unet(nn.Module):
    def __init__(self):
    # Now here write a simple network that takes input a time_step value and x_t
    # and predicts \epsilon_thetha and so i want to write a normal convolutional network
    # and some dropout and normalisation to do this and output 
        super().__init__()
        self.time_dim = 64
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_dim, self.time_dim),
            nn.ReLU(),
            nn.Linear(self.time_dim, self.time_dim)
        )
        self.in_conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU()
        )
        
        # Encoder (downsampling blocks)
        self.down1 = nn.Sequential(  # To 64 -> 128, pool to 14x14
            nn.Conv2d(64, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.down2 = nn.Sequential(  # To 128 -> 256, pool to 7x7
            nn.Conv2d(128, 128, 3, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Bottleneck (at 7x7)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU()
        )
        
        # Decoder (upsampling blocks with skips and dropout)
        self.up1 = nn.Sequential(  # Upsample 256->128, concat with skip (256 in), to 128
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),  # To 14x14
            nn.Conv2d(256, 128, 3, padding=1),  # After concat
            nn.GroupNorm(8, 128),
            nn.ReLU(),
            nn.Dropout2d(0.1)
        )
        
        self.up2 = nn.Sequential(  # Upsample 128->64, concat with skip (128 in), to 64
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),  # To 28x28
            nn.Conv2d(128, 64, 3, padding=1),  # After concat
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.Dropout2d(0.1)
        )
        
        # Output conv (predict noise)
        self.out_conv = nn.Conv2d(64, 1, kernel_size=1)
        self.time_proj1 = nn.Linear(self.time_dim, 64)
        self.time_proj2 = nn.Linear(self.time_dim, 128)
        self.time_proj3 = nn.Linear(self.time_dim, 256)
        self.time_proj_up1 = nn.Linear(self.time_dim, 128)
        self.time_proj_up2 = nn.Linear(self.time_dim, 64)

        pass
    def sinusoidal_embedding(self,timestep):
        device = timestep.device
        half_dim = self.time_dim // 2
        embeddings = torch.log(torch.tensor(10000)) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = timestep[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
    def forward(self, x, t):
        batch_size = x.shape[0]

        # Embed timestep
        t_emb = self.sinusoidal_embedding(t)        # [batch, 64]
        t_emb = self.time_mlp(t_emb)                # [batch, 64]

        # Project to match layers
        t1 = self.time_proj1(t_emb).view(batch_size, 64, 1, 1)
        t2 = self.time_proj2(t_emb).view(batch_size, 128, 1, 1)
        t3 = self.time_proj3(t_emb).view(batch_size, 256, 1, 1)
        tu1 = self.time_proj_up1(t_emb).view(batch_size, 128, 1, 1)
        tu2 = self.time_proj_up2(t_emb).view(batch_size, 64, 1, 1)

        # Encoder
        x1 = self.in_conv(x)
        x1 = x1 + t1
        x1 = F.relu(x1)

        x2 = self.down1(x1)
        x2 = x2 + t2
        x2 = F.relu(x2)

        x3 = self.down2(x2)
        x3 = x3 + t3
        x3 = F.relu(x3)

        # Bottleneck
        x_b  = self.bottleneck(x3)

        # Decoder
        x = self.up1[0](x_b)                 # [batch, 128, 14, 14]
        x = torch.cat([x, x2], dim=1)        # [batch, 256, 14, 14]
        x = self.up1[1:](x)                  # [batch, 128, 14, 14]
        x = x + tu1
        x = F.relu(x)

        x = self.up2[0](x)                   # [batch, 64, 28, 28]
        x = torch.cat([x, x1], dim=1)        # [batch, 128, 28, 28]
        x = self.up2[1:](x)                  # [batch, 64, 28, 28]
        x = x + tu2
        x = F.relu(x)

        # Output
        epsilon_theta = self.out_conv(x)     # [batch, 1, 28, 28]
        return epsilon_theta



class D3PM(nn.Module):
    def __init__(self): 
        super().__init__()
        

class ConditionalD3PM(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        

class DDPM(nn.Module):
    def __init__(self,train_loader,test_loader,run_name,learning_rate, epochs,batch_size,device,beta_start,beta_end):
        super().__init__()
        self.Noise = NoiseSchedulerDDPM(1000,"linear",beta_start=beta_start,beta_end=beta_end)
        self.train_loader = train_loader
        self.test_loader  = test_loader
        self.run_name = run_name
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.model = Unet().to(device)
        
    def forward_process(self,x0,device):
        batch_size = x0.shape[0]
        t = torch.randint(0,self.Noise.num_timesteps,(batch_size,),device=device).long()
        noise = torch.randn_like(x0).to(device)
        sqrt_alphas = self.Noise.sqrt_alphas[t].view(-1,1,1,1)
        sqrt_one_minus_alphas = self.Noise.sqrt_one_minus_alphas[t].view(-1,1,1,1)
        xt = sqrt_alphas*x0 + sqrt_one_minus_alphas*noise
        return xt,t,noise

    def get_mu_theta(self, x, t, epsilon_theta):
        """
        Computes the reverse process mean μ_θ(x_t, t) from Eq. 11.
        """
        alpha_t = self.Noise.alphas[t].to(self.device)
        beta_t = self.Noise.betas[t].to(self.device)
        alpha_bar_t = self.Noise.alphas_cumprod[t].to(self.device)
        mu_theta = (1 / torch.sqrt(alpha_t)) * (x - (beta_t / torch.sqrt(1 - alpha_bar_t)) * epsilon_theta)
        return mu_theta

    def reverse_process(self, device, num_samples=16, num_steps=None):
        """
        Implements Algorithm 2: Denoising from x_T ~ N(0, I) to x_0.
        Args:
            device: torch.device
            num_samples: int, number of samples to generate (batch size)
            num_steps: int, number of denoising steps (default: 1000)
        Returns:
            x_0: torch.Tensor, shape (num_samples, 1, 28, 28)
        """
        if num_steps is None:
            num_steps = self.Noise.num_timesteps
        self.model.eval()
        with torch.no_grad():
            # Start from pure noise x_T
            x = torch.randn(num_samples, 1, 28, 28, device=device)
            for t_val in range(num_steps, 0, -1):  # t from T down to 1
                t_tensor = torch.full((num_samples,), t_val, device=device, dtype=torch.long)
                # Predict noise ε_θ(x_t, t)
                epsilon_theta = self.model(x, t_tensor)
                # Compute μ_θ (Eq. 11)
                mu_theta = self.get_mu_theta(x, t_tensor, epsilon_theta)
                # Variance σ_t^2 = β_t
                sigma_t = torch.sqrt(self.Noise.betas[t_val].to(device))
                # z ~ N(0, I) if t > 1, else 0
                z = torch.randn_like(x, device=device) if t_val > 1 else 0
                # x_{t-1} = μ_θ + σ_t * z
                x = mu_theta + sigma_t * z
            # At t=0, final denoising step (no noise added)
            t_tensor = torch.zeros((num_samples,), device=device, dtype=torch.long)
            epsilon_theta = self.model(x, t_tensor)
            mu_theta = self.get_mu_theta(x, t_tensor, epsilon_theta)
            x = mu_theta  # No z at t=0
        # Clamp to [0,1] for valid images
        x = torch.clamp(x, 0, 1)
        return x

    def train_model(self):
        """
        Training loop using simplified MSE objective (Eq. 14).
        Saves checkpoints every 5 epochs and final model to self.run_name.
        """
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        os.makedirs(self.run_name, exist_ok=True)
        
        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0
            progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.epochs}')
            
            for batch_idx, (data, _) in enumerate(progress_bar):
                data = data.to(self.device)  # [batch_size, 1, 28, 28]
                optimizer.zero_grad()
                # Forward process: get xt, t, true noise
                xt, t, noise = self.forward_process(data, self.device)
                # Predict noise with U-Net
                epsilon_theta = self.model(xt, t)
                # MSE loss: ||ε - ε_θ||^2
                loss = F.mse_loss(epsilon_theta, noise)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
            
            avg_train_loss = running_loss / len(self.train_loader)
            print(f'Epoch {epoch+1}/{self.epochs} - Average Train Loss: {avg_train_loss:.4f}')
            
            # Optional validation every 5 epochs
            # if (epoch + 1) % 5 == 0:
            #     self.model.eval()
            #     with torch.no_grad():
            #         # Get one test batch
            #         test_data, _ = next(iter(self.test_loader))
            #         test_data = test_data.to(self.device)
            #         # Generate pure noise and denoise (short steps for speed)
            #         x_T = torch.randn_like(test_data)
            #         x_0_recon = self.reverse_process(self.device, num_samples=test_data.shape[0], num_steps=50)
            #         recon_loss = F.mse_loss(x_0_recon, test_data)
            #         print(f'  Validation Reconstruction MSE: {recon_loss.item():.4f}')
                
            #     # Save checkpoint
            #     torch.save(self.model.state_dict(), f"{self.run_name}/model_epoch_{epoch+1}.pth")
        
        # Save final model
        torch.save(self.model.state_dict(), f"{self.run_name}/model.pth")
        print(f"Training complete. Final model saved to {self.run_name}/model.pth")

    def load_state_dict(self, path):
        """
        Load the U-Net state dict from file.
        """
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Model loaded from {path}")

class ConditionalDDPM(nn.Module):
    def __init__(self, num_classes): 
        super().__init__()
        self.num_classes = num_classes