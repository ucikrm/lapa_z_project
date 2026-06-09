import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallEncoder(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(6, 32, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.fc = nn.Linear(256, latent_dim)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        h = self.net(x).flatten(1)
        z = self.fc(h)
        return z


class SmallDecoder(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()

        self.x1_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        self.z_fc = nn.Linear(latent_dim, 128)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x1, z):
        # Stop-gradient on x1 path, similar motivation as LAPA phase-1.
        x1_feat = self.x1_encoder(x1).detach()

        b, c, h, w = x1_feat.shape
        z_feat = self.z_fc(z).view(b, 128, 1, 1).expand(b, 128, h, w)

        hcat = torch.cat([x1_feat, z_feat], dim=1)
        x2_hat = self.decoder(hcat)
        return x2_hat


class VectorQuantizer(nn.Module):
    def __init__(self, num_codes=64, latent_dim=32, beta=0.25):
        super().__init__()
        self.beta = beta
        self.codebook = nn.Embedding(num_codes, latent_dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / num_codes, 1.0 / num_codes)

    def forward(self, z_e):
        distances = (
            z_e.pow(2).sum(dim=1, keepdim=True)
            - 2 * z_e @ self.codebook.weight.t()
            + self.codebook.weight.pow(2).sum(dim=1).unsqueeze(0)
        )

        indices = torch.argmin(distances, dim=1)
        z_q = self.codebook(indices)

        codebook_loss = F.mse_loss(z_q, z_e.detach())
        commitment_loss = F.mse_loss(z_e, z_q.detach())
        vq_loss = codebook_loss + self.beta * commitment_loss

        z_q_st = z_e + (z_q - z_e).detach()

        return z_q_st, indices, vq_loss


class VQPhase1Model(nn.Module):
    def __init__(self, latent_dim=32, num_codes=64):
        super().__init__()
        self.encoder = SmallEncoder(latent_dim=latent_dim)
        self.vq = VectorQuantizer(num_codes=num_codes, latent_dim=latent_dim)
        self.decoder = SmallDecoder(latent_dim=latent_dim)

    def forward(self, x1, x2):
        z_e = self.encoder(x1, x2)
        z_q, indices, vq_loss = self.vq(z_e)
        x2_hat = self.decoder(x1, z_q)

        return {
            "x2_hat": x2_hat,
            "z": z_q,
            "z_e": z_e,
            "indices": indices,
            "vq_loss": vq_loss,
        }


class ContinuousPhase1Model(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.encoder = SmallEncoder(latent_dim=latent_dim)
        self.norm = nn.LayerNorm(latent_dim)
        self.decoder = SmallDecoder(latent_dim=latent_dim)

    def forward(self, x1, x2):
        z = self.encoder(x1, x2)
        z = self.norm(z)
        x2_hat = self.decoder(x1, z)
        return {"x2_hat": x2_hat, "z": z}


def vicreg_var_cov_loss(z, gamma=1.0, eps=1e-4):
    z = z.float()
    z_centered = z - z.mean(dim=0)

    std = torch.sqrt(z_centered.var(dim=0) + eps)
    var_loss = torch.mean(torch.relu(gamma - std))

    b, d = z.shape
    if b <= 1:
        cov_loss = torch.tensor(0.0, device=z.device)
    else:
        cov = (z_centered.T @ z_centered) / (b - 1)
        off_diag = cov - torch.diag(torch.diag(cov))
        cov_loss = (off_diag ** 2).sum() / d

    return var_loss, cov_loss


class NaiveContinuousPhase1Model(nn.Module):
    """
    Continuous latent action model with no anti-collapse regularization.
    Trained only with reconstruction loss.

    This is the collapse-prone baseline:
        x_t, x_{t+1} -> z -> decoder(x_t, z) -> xhat_{t+1}
    """
    def __init__(self, latent_dim=32):
        super().__init__()
        self.encoder = SmallEncoder(latent_dim=latent_dim)
        self.decoder = SmallDecoder(latent_dim=latent_dim)

    def forward(self, x1, x2):
        z = self.encoder(x1, x2)
        x2_hat = self.decoder(x1, z)
        return {
            "x2_hat": x2_hat,
            "z": z,
        }
