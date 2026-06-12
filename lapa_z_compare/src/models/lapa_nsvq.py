import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseLatentModel, CommonEncoder, CommonDecoder

class NSVQQuantizer(nn.Module):
    """
    Noise-Substitution Vector Quantization (NSVQ) Module.
    During training, replaces the quantization error with a Gaussian noise
    matching the statistics of the quantization error.
    During evaluation, performs standard nearest-neighbor vector quantization.
    """
    def __init__(self, num_codes=64, latent_dim=32, beta=0.25):
        super().__init__()
        self.num_codes = num_codes
        self.latent_dim = latent_dim
        self.beta = beta
        
        self.codebook = nn.Embedding(num_codes, latent_dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / num_codes, 1.0 / num_codes)

    def forward(self, z_e):
        # Compute pairwise Euclidean distances
        distances = (
            z_e.pow(2).sum(dim=-1, keepdim=True)
            - 2 * z_e @ self.codebook.weight.t()
            + self.codebook.weight.pow(2).sum(dim=-1).unsqueeze(0)
        )

        indices = torch.argmin(distances, dim=-1)
        z_q = self.codebook(indices)

        # Standard VQ VAE commitment loss (pulls encoder towards codebook)
        # The codebook itself receives gradients directly from reconstruction loss
        # via the differentiable noise term during training.
        commitment_loss = F.mse_loss(z_e, z_q.detach())
        vq_loss = self.beta * commitment_loss

        if self.training:
            # Quantization error vector
            q_err = z_q - z_e
            # Per-sample quantization-residual norm (safe from NaNs)
            q_err_norm = torch.sqrt(torch.sum(q_err**2, dim=-1, keepdim=True) + 1e-8)
            
            # Generate unit-norm random noise
            noise = torch.randn_like(z_e)
            noise_norm = torch.sqrt(torch.sum(noise**2, dim=-1, keepdim=True) + 1e-8)
            normalized_noise = noise / noise_norm
            
            # Substituted output: z_out = z_e + ||z_q - z_e|| * (v / ||v||)
            # Both terms (z_e and the error norm) are differentiable, allowing gradients
            # to flow back to the encoder (via z_e and norm) and the codebook (via z_q in norm).
            z_out = z_e + q_err_norm * normalized_noise
        else:
            # Inference: use exact quantized vectors
            z_out = z_q

        return z_out, indices, vq_loss


class StandardVQQuantizer(nn.Module):
    """
    Standard Vector Quantization (VQ) Module using Straight-Through Estimator (STE).
    """
    def __init__(self, num_codes=64, latent_dim=32, beta=0.25):
        super().__init__()
        self.num_codes = num_codes
        self.latent_dim = latent_dim
        self.beta = beta
        
        self.codebook = nn.Embedding(num_codes, latent_dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / num_codes, 1.0 / num_codes)

    def forward(self, z_e):
        # Compute pairwise Euclidean distances
        distances = (
            z_e.pow(2).sum(dim=-1, keepdim=True)
            - 2 * z_e @ self.codebook.weight.t()
            + self.codebook.weight.pow(2).sum(dim=-1).unsqueeze(0)
        )

        indices = torch.argmin(distances, dim=-1)
        z_q = self.codebook(indices)

        # Standard VQ VAE losses: codebook MSE and commitment MSE
        codebook_loss = F.mse_loss(z_q, z_e.detach())
        commitment_loss = F.mse_loss(z_e, z_q.detach())
        vq_loss = codebook_loss + self.beta * commitment_loss

        # Straight-through estimator
        if self.training:
            z_out = z_e + (z_q - z_e).detach()
        else:
            z_out = z_q

        return z_out, indices, vq_loss


class LapaNSVQModel(BaseLatentModel):
    def __init__(self, latent_dim=32, num_codes=64, beta=0.25):
        super().__init__()
        self.encoder = CommonEncoder(latent_dim=latent_dim)
        self.quantizer = NSVQQuantizer(num_codes=num_codes, latent_dim=latent_dim, beta=beta)
        self.decoder = CommonDecoder(latent_dim=latent_dim)

    def encode(self, x_t, x_tpH, label=None):
        z_e = self.encoder(x_t, x_tpH)
        z_out, indices, _ = self.quantizer(z_e)
        return z_out

    def decode(self, x_t, z, label=None):
        return self.decoder(x_t, z)

    def forward(self, x_t, x_tpH, label=None, structural=None):
        z_e = self.encoder(x_t, x_tpH)
        z_out, indices, vq_loss = self.quantizer(z_e)
        x_hat = self.decoder(x_t, z_out)

        rec_loss = F.mse_loss(x_hat, x_tpH)
        total_loss = rec_loss + vq_loss

        return {
            "z": z_out,
            "x_hat": x_hat,
            "indices": indices,
            "loss_dict": {
                "loss": total_loss,
                "rec_loss": rec_loss,
                "vq_loss": vq_loss
            }
        }
