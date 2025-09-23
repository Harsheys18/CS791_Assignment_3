import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, groups=8):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(groups, out_channels)
        self.relu = nn.SiLU()  # SiLU (Swish) is common in diffusion models
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)

    def forward(self, x, t_emb):
        h = self.conv(x)
        # Add time embedding
        t_emb = self.time_mlp(t_emb)[:, :, None, None]  # Shape: (batch, out_channels, 1, 1)
        h = h + t_emb
        h = self.norm(h)
        h = self.relu(h)
        return h

class SelfAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.out = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(b, 3 * c, h * w)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(c))
        attn = torch.einsum('bci,bcj->bij', q * scale, k * scale)
        attn = F.softmax(attn, dim=-1)
        h = torch.einsum('bij,bcj->bci', attn, v).reshape(b, c, h, w)
        h = self.out(h)
        return x + h  # Residual connection

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_channels=64, time_emb_dim=128):
        super().__init__()
        self.time_emb_dim = time_emb_dim

        # Time embedding
        self.time_embed = SinusoidalPositionEmbedding(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        # Encoder
        self.enc1 = ConvBlock(in_channels, base_channels, time_emb_dim)
        self.enc2 = ConvBlock(base_channels, base_channels * 2, time_emb_dim)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4, time_emb_dim)
        self.pool = nn.MaxPool2d(2)

        # Self-attention at 16x16 (assuming input is 32x32)
        self.attn = SelfAttention(base_channels * 2)

        # Bottleneck
        self.bottleneck = ConvBlock(base_channels * 4, base_channels * 8, time_emb_dim)

        # Decoder
        self.dec1 = ConvBlock(base_channels * 12, base_channels * 4, time_emb_dim)
        self.dec2 = ConvBlock(base_channels * 6, base_channels * 2, time_emb_dim)
        self.dec3 = ConvBlock(base_channels * 3, base_channels, time_emb_dim)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Final convolution
        self.final = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x, t):
        # Time embedding
        t_emb = self.time_embed(t)
        t_emb = self.time_mlp(t_emb)

        # Encoder
        e1 = self.enc1(x, t_emb)
        e2 = self.pool(e1)
        e2 = self.enc2(e2, t_emb)
        e2 = self.attn(e2)  # Self-attention at 16x16
        e3 = self.pool(e2)
        e3 = self.enc3(e3, t_emb)
        b = self.pool(e3)
        b = self.bottleneck(b, t_emb)

        # Decoder with skip connections
        d1 = self.up(b)
        d1 = torch.cat([d1, e3], dim=1)  # Skip connection
        d1 = self.dec1(d1, t_emb)
        d2 = self.up(d1)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2, t_emb)
        d3 = self.up(d2)
        d3 = torch.cat([d3, e1], dim=1)
        d3 = self.dec3(d3, t_emb)

        # Output
        out = self.final(d3)
        return out

# Example usage with your DDPM class
class NoiseSchedule:
    def __init__(self, num_timesteps, beta_start=1e-4, beta_end=0.02):
        self.num_timesteps = num_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas = torch.sqrt(self.alphas)
        self.sqrt_one_minus_alphas = torch.sqrt(1 - self.alpha_bars)

class DDPM(nn.Module):
    def __init__(self, noise_schedule, model):
        super().__init__()
        self.Noise = noise_schedule
        self.model = model

    def forward_process(self, x0, device):
        batch_size = x0.shape[0]
        t = torch.randint(0, self.Noise.num_timesteps, (batch_size,), device=device).long()
        noise = torch.randn_like(x0).to(device)
        sqrt_alphas = self.Noise.sqrt_alphas[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas = self.Noise.sqrt_one_minus_alphas[t].view(-1, 1, 1, 1)
        xt = sqrt_alphas * x0 + sqrt_one_minus_alphas * noise
        return xt, t, noise

    def reverse_process(self, xt, t, device):
        t = torch.tensor(t, device=device, dtype=torch.long) if isinstance(t, int) else t
        epsilon_theta = self.model(xt, t)
        beta_t = self.Noise.betas[t].view(-1, 1, 1, 1)
        alpha_t = self.Noise.alphas[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar_t = self.Noise.sqrt_one_minus_alphas[t].view(-1, 1, 1, 1)
        mu_theta = (1 / torch.sqrt(alpha_t)) * (xt - (beta_t / sqrt_one_minus_alpha_bar_t) * epsilon_theta)
        sigma_t = torch.sqrt(beta_t)
        z = torch.randn_like(xt).to(device) if (t > 1).any() else torch.zeros_like(xt).to(device)
        x_t_minus_1 = mu_theta + sigma_t * z
        return x_t_minus_1

    def compute_loss(self, x0, device):
        xt, t, noise = self.forward_process(x0, device)
        epsilon_theta = self.model(xt, t)
        loss = torch.mean((noise - epsilon_theta) ** 2)
        return loss

# Instantiate and test
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
noise_schedule = NoiseSchedule(num_timesteps=1000)
unet = UNet(in_channels=3, out_channels=3, base_channels=64, time_emb_dim=128).to(device)
ddpm = DDPM(noise_schedule, unet)
x0 = torch.rand(16, 3, 32, 32).to(device) * 2 - 1  # Batch of images in [-1, 1]
loss = ddpm.compute_loss(x0, device)
xt, t, _ = ddpm.forward_process(x0, device)
x_t_minus_1 = ddpm.reverse_process(xt, t, device)