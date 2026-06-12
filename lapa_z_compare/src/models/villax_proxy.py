import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseLatentModel, CommonEncoder, CommonDecoder
from .lapa_nsvq import StandardVQQuantizer

class StructuralFDM(nn.Module):
    """
    Predicts the future structural state s_{t+H} given the current structural state s_t
    and the action latent z.
    """
    def __init__(self, latent_dim=32, struct_channels=1):
        super().__init__()
        self.z_fc = nn.Linear(latent_dim, 32 * 28 * 28)
        self.conv = nn.Sequential(
            nn.Conv2d(struct_channels + 32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, struct_channels, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, s_t, z):
        b = s_t.shape[0]
        z_feat = self.z_fc(z).view(b, 32, 28, 28)
        h = torch.cat([s_t, z_feat], dim=1)
        s_tpH_hat = self.conv(h)
        return s_tpH_hat


class VillaXProxyModel(BaseLatentModel):
    """
    villa-X-inspired Structure-Grounded Latent Action Proxy.
    Trains with visual reconstruction plus auxiliary Sobel edge dynamics prediction.
    """
    def __init__(self, latent_dim=32, lambda_s=1.0, beta=0.25, num_codes=64):
        super().__init__()
        self.encoder = CommonEncoder(latent_dim=latent_dim)
        self.quantizer = StandardVQQuantizer(num_codes=num_codes, latent_dim=latent_dim, beta=beta)
        self.decoder = CommonDecoder(latent_dim=latent_dim)
        self.fdm_s = StructuralFDM(latent_dim=latent_dim, struct_channels=1)
        self.lambda_s = lambda_s

    def compute_sobel_edges(self, img):
        # Convert RGB to Grayscale
        gray = img.mean(dim=1, keepdim=True) # [B, 1, H, W]
        
        # Sobel filters
        kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], dtype=img.dtype, device=img.device).view(1, 1, 3, 3)
        ky = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], dtype=img.dtype, device=img.device).view(1, 1, 3, 3)
        
        gx = F.conv2d(gray, kx, padding=1)
        gy = F.conv2d(gray, ky, padding=1)
        
        edge = torch.sqrt(gx**2 + gy**2 + 1e-6)
        # Downsample to structural grid resolution (28x28)
        edge_low = F.interpolate(edge, size=(28, 28), mode="area")
        # Normalize to [0, 1] range for stability (per-sample)
        b = edge_low.shape[0]
        edge_flat = edge_low.view(b, -1)
        m_min = edge_flat.min(dim=1, keepdim=True).values.view(b, 1, 1, 1)
        m_max = edge_flat.max(dim=1, keepdim=True).values.view(b, 1, 1, 1)
        edge_norm = (edge_low - m_min) / (m_max - m_min + 1e-6)
        return edge_norm

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

        # Compute structural proxy (Sobel edge maps)
        s_t = self.compute_sobel_edges(x_t)
        s_tpH = self.compute_sobel_edges(x_tpH)
        
        # Predict future structural state
        s_tpH_hat = self.fdm_s(s_t, z_out)

        rec_loss = F.mse_loss(x_hat, x_tpH)
        struct_loss = F.mse_loss(s_tpH_hat, s_tpH)
        total_loss = rec_loss + self.lambda_s * struct_loss + vq_loss

        return {
            "z": z_out,
            "x_hat": x_hat,
            "s_tpH_hat": s_tpH_hat,
            "indices": indices,
            "loss_dict": {
                "loss": total_loss,
                "rec_loss": rec_loss,
                "struct_loss": struct_loss,
                "vq_loss": vq_loss
            }
        }
