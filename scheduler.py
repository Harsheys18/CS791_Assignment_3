import torch
import math

class NoiseSchedulerDDPM():
    """
    Noise scheduler for the DDPM model

    Args:
        num_timesteps: int, the number of timesteps
        type: str, the type of scheduler to use
        **kwargs: additional arguments for the scheduler

    This object sets up all the constants like alpha, beta, sigma, etc. required for the DDPM model
    
    """
    def __init__(self, num_timesteps=50, type="linear", **kwargs):

        self.num_timesteps = num_timesteps
        self.type = type

        if type == "linear":
            self.init_linear_schedule(**kwargs)
        else:
            raise NotImplementedError(f"{type} scheduler is not implemented") # change this if you implement additional schedulers


    def init_linear_schedule(self, beta_start, beta_end):
        """
        Precompute whatever quantities are required for training and sampling
        """

        self.betas = torch.linspace(beta_start, beta_end, self.num_timesteps, dtype=torch.float32)

        self.alphas = None

    def __len__(self):
        return self.num_timesteps
class MaskSchedulerD3PM:
    """
    Mask scheduler for Discrete Diffusion (D3PM) models.
    """

    def __init__(self, num_timesteps=50, mask_type="linear", **kwargs):
        self.num_timesteps = num_timesteps
        self.mask_type = mask_type

        if mask_type == "linear":
            start_prob = kwargs.get("start_prob", 1e-3)
            end_prob   = kwargs.get("end_prob", 0.9)   # allow more corruption
            self.mask_probs = torch.linspace(start_prob, end_prob, num_timesteps)

        elif mask_type == "uniform":
            prob = kwargs.get("mask_prob", 0.25)
            self.mask_probs = torch.full((num_timesteps,), prob)

        elif mask_type == "cosine":
            # Cosine schedule (like beta schedule in DDPMs)
            steps = torch.arange(0, num_timesteps + 1, dtype=torch.float32)
            alphas_cumprod = torch.cos(((steps / num_timesteps) + 0.008) / 1.008 * math.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            self.mask_probs = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])

        elif mask_type == "quadratic":
            start_prob = kwargs.get("start_prob", 1e-3)
            end_prob   = kwargs.get("end_prob", 0.9)
            t = torch.linspace(0, 1, num_timesteps)
            self.mask_probs = start_prob + (end_prob - start_prob) * (t ** 2)

        elif mask_type == "exponential":
            start_prob = kwargs.get("start_prob", 1e-3)
            end_prob   = kwargs.get("end_prob", 0.9)
            t = torch.linspace(0, 1, num_timesteps)
            self.mask_probs = start_prob * (end_prob / start_prob) ** t

        else:
            raise NotImplementedError(f"{mask_type} mask scheduler is not implemented")

    def __len__(self):
        return self.num_timesteps

    def __getitem__(self, t):
        """
        Get mask probability for timestep t (1-indexed).
        """
        if isinstance(t, torch.Tensor):
            t = t.to(self.mask_probs.device)
            return self.mask_probs[t - 1]
        else:
            return self.mask_probs[t - 1]
