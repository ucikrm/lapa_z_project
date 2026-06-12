import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseLatentModel, CommonEncoder, CommonDecoder
from ..losses.vicreg_loss import vicreg_loss

class VicregContinuousModel(BaseLatentModel):
    """
    VICReg Continuous Latent Action Model.
    Trained with reconstruction loss plus batch-wise variance/covariance losses.
    """
    def __init__(self, latent_dim=32, lambda_var=1.0, lambda_cov=0.05):
        super().__init__()
        self.encoder = CommonEncoder(latent_dim=latent_dim)
        self.decoder = CommonDecoder(latent_dim=latent_dim)
        self.lambda_var = lambda_var
        self.lambda_cov = lambda_cov

    def encode(self, x_t, x_tpH, label=None):
        return self.encoder(x_t, x_tpH)

    def decode(self, x_t, z, label=None):
        return self.decoder(x_t, z)

    def forward(self, x_t, x_tpH, label=None, structural=None):
        z = self.encoder(x_t, x_tpH)
        x_hat = self.decoder(x_t, z)

        rec_loss = F.mse_loss(x_hat, x_tpH)
        var_loss, cov_loss = vicreg_loss(z)
        
        total_loss = rec_loss + self.lambda_var * var_loss + self.lambda_cov * cov_loss

        return {
            "z": z,
            "x_hat": x_hat,
            "loss_dict": {
                "loss": total_loss,
                "rec_loss": rec_loss,
                "var_loss": var_loss,
                "cov_loss": cov_loss
            }
        }
