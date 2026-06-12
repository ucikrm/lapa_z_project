import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseLatentModel, CommonDecoder

class VAEEncoder(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(6, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        h = self.net(x).flatten(1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class AdaWorldBVAEModel(BaseLatentModel):
    def __init__(self, latent_dim=32, beta=2e-4):
        super().__init__()
        self.encoder = VAEEncoder(latent_dim=latent_dim)
        self.decoder = CommonDecoder(latent_dim=latent_dim)
        self.beta = beta

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def encode(self, x_t, x_tpH, label=None):
        mu, logvar = self.encoder(x_t, x_tpH)
        return mu

    def decode(self, x_t, z, label=None):
        return self.decoder(x_t, z)

    def forward(self, x_t, x_tpH, label=None, structural=None):
        mu, logvar = self.encoder(x_t, x_tpH)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(x_t, z)

        rec_loss = F.mse_loss(x_hat, x_tpH)
        
        # KL Divergence: sum over dims, mean over batch
        kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        
        total_loss = rec_loss + self.beta * kl_loss

        return {
            "z": z,
            "x_hat": x_hat,
            "mu": mu,
            "logvar": logvar,
            "loss_dict": {
                "loss": total_loss,
                "rec_loss": rec_loss,
                "kl_loss": kl_loss
            }
        }
