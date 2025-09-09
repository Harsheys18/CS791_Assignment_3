import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F

class D3PM(nn.Module):
    def __init__(self, vocab_size=256, mask_token=255, hidden_dim=512): # Add any required parameters
        super().__init__()
        # Define your conditional model architecture here
        self.vocab_size = vocab_size
        self.mask_token = mask_token

        # Denoising network (classifier)
        self.denoising_net = nn.Sequential(
            nn.Linear(28*28, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 28*28*vocab_size)   # logits over discrete tokens
        )
        
    def noising(self, x, t, beta_t):
        """
        Forward process: q(x_t | x_{t-1})
        Args:
            x: tensor of shape (B, 28, 28), with values in {0,...,255}
            t: timestep (int or tensor)
            beta_t: noise rate for timestep t (float in [0,1])
        Returns:
            x_t: noisy version of x with [MASK] applied
        """
        device = x.device
        beta_t = beta_t.to(device)              # 🔑 move to same device
        mask = torch.rand_like(x.float(), device=device) < beta_t
        x_t = x.clone()
        x_t[mask] = self.mask_token
        return x_t


    def denoising(self, x_t, t):
        """
        Reverse process: predict distribution over x_0 given x_t
        Args:
            x_t: noisy input, shape (B, 28, 28), values in {0,...,255 or mask}
            t: timestep (unused here, but normally fed as embedding)
        Returns:
            logits: (B, 28*28, vocab_size)
        """
        B = x_t.shape[0]
        x_flat = x_t.view(B, -1).float() / 255.0  
        logits = self.denoising_net(x_flat)       
        logits = logits.view(B, 28*28, self.vocab_size)
        return logits

class ConditionalD3PM(nn.Module):
    def __init__(self, num_classes=10, vocab_size=256, mask_token=255, hidden_dim=512): # Add any required parameters
        super().__init__()
        # Define your model architecture here
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.mask_token = mask_token

        # MLP denoiser with class conditioning via concatenation
        self.denoising_net = nn.Sequential(
            nn.Linear(28*28 + num_classes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 28*28 * vocab_size)
        )

    def noising(self, x, t, beta_t):
        """
        Forward process: q(x_t | x_{t-1})
        Args:
            x: (B, 28, 28), int tokens {0,...,255}
            t: timestep (unused except for scheduler)
            beta_t: mask probability (broadcastable to x)
        Returns:
            x_t: noisy x with mask applied
        """
        device = x.device
        beta_t = beta_t.to(device)
        mask = torch.rand_like(x.float(), device=device) < beta_t
        x_t = x.clone()
        x_t[mask] = self.mask_token
        return x_t

    def denoising(self, x_t, t, y):
        """
        Reverse process: p(x_{t-1} | x_t, y)
        Args:
            x_t: (B, 28, 28), int tokens
            t: (B,) timesteps (unused here)
            y: (B,) class labels
        Returns:
            logits: (B, 28*28, vocab_size)
        """
        B = x_t.shape[0]
        x_flat = x_t.view(B, -1).float() / 255.0    # normalize to [0,1]
        
        # one-hot encode labels and concatenate
        y_onehot = F.one_hot(y, num_classes=self.num_classes).float()
        x_input = torch.cat([x_flat, y_onehot], dim=1)  # (B, 784+num_classes)

        logits = self.denoising_net(x_input)             # (B, 784*vocab)
        logits = logits.view(B, 28*28, self.vocab_size)
        return logits

class DDPM(nn.Module):
    def __init__(self): # Add any required parameters
        super().__init__()
        # Define your model architecture here

class ConditionalDDPM(nn.Module):
    def __init__(self, num_classes): # Add any required parameters
        super().__init__()
        self.num_classes = num_classes
        # Define your conditional model architecture here