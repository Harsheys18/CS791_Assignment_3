import torch

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
    
import torch

class MaskSchedulerD3PM:
    """
    Mask scheduler for Discrete Diffusion (D3PM) models.

    Args:
        num_timesteps: int, number of timesteps in the diffusion process
        mask_type: str, type of mask scheduling ("uniform", "linear", etc.)
        **kwargs: additional arguments for mask scheduling
    """

    def __init__(self, num_timesteps=50, mask_type="linear", **kwargs):
        self.num_timesteps = num_timesteps
        self.mask_type = mask_type

        if mask_type == "linear":
            start_prob = kwargs.get("start_prob", 1e-3)   # small noise at t=1
            end_prob   = kwargs.get("end_prob", 0.5)      # max mask probability
            self.init_linear_schedule(start_prob, end_prob)
        elif mask_type == "uniform":
            prob = kwargs.get("mask_prob", 0.25)          # fixed mask probability
            self.mask_probs = torch.full((num_timesteps,), prob)
        else:
            raise NotImplementedError(f"{mask_type} mask scheduler is not implemented")

    def init_linear_schedule(self, start_prob, end_prob):
        """
        Initializes a linear mask schedule where the mask probability increases linearly.
        """
        self.mask_probs = torch.linspace(start_prob, end_prob, self.num_timesteps)

    def __len__(self):
        return self.num_timesteps

    def __getitem__(self, t):
        """
        Get mask probability for timestep t (1-indexed like in training loop).
        """
        return self.mask_probs[(t-1).to(self.mask_probs.device)]
