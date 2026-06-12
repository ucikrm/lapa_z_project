import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

def gather_all_tensors(tensor):
    """
    Gathers a tensor from all GPUs in DDP mode to compute global batch statistics.
    """
    if not dist.is_available() or not dist.is_initialized():
        return tensor
    
    world_size = dist.get_world_size()
    gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, tensor)
    return torch.cat(gathered_tensors, dim=0)

def vicreg_loss(z, gamma=1.0, eps=1e-4):
    """
    Computes global batch-wise VICReg variance and covariance losses.
    """
    # Gather latents from all GPUs if using DDP
    z_global = gather_all_tensors(z)
    
    # Center the latents
    z_centered = z_global - z_global.mean(dim=0)
    
    # 1. Variance Loss: pushes standard deviation along each dimension to be >= gamma
    std = torch.sqrt(z_centered.var(dim=0, unbiased=False) + eps)
    var_loss = torch.mean(torch.relu(gamma - std))
    
    # 2. Covariance Loss: pushes off-diagonal covariance values to 0 (decorrelation)
    b, d = z_global.shape
    if b <= 1:
        cov_loss = torch.tensor(0.0, device=z.device)
    else:
        cov = (z_centered.T @ z_centered) / (b - 1)
        off_diag = cov - torch.diag(torch.diag(cov))
        cov_loss = (off_diag ** 2).sum() / d
        
    return var_loss, cov_loss
