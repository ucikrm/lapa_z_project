import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseLatentModel(nn.Module):
    """
    Common interface for all comparative latent action models.
    """
    def forward(self, x_t, x_tpH, label=None, structural=None):
        raise NotImplementedError

    def encode(self, x_t, x_tpH, label=None):
        raise NotImplementedError

    def decode(self, x_t, z, label=None):
        raise NotImplementedError


class CommonEncoder(nn.Module):
    """
    Standard encoder E(x_t, x_{t+H}) -> z.
    Concats two frames along channel dimension and downsamples to a latent vector.
    """
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
        self.fc = nn.Linear(256, latent_dim)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        h = self.net(x).flatten(1)
        z = self.fc(h)
        return z


class CommonDecoder(nn.Module):
    """
    Standard decoder D(x_t, z) -> \hat{x}_{t+H}.
    Reconstructs the target frame using appearance encoding of x_t and latent action z.
    The appearance encoder is trained jointly (no detaching) to resolve the untrained appearance encoder flaw.
    """
    def __init__(self, latent_dim=32):
        super().__init__()
        self.x1_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.z_fc = nn.Linear(latent_dim, 128)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x1, z):
        # Appearance features (no detach, trained jointly)
        x1_feat = self.x1_encoder(x1)

        b, c, h, w = x1_feat.shape
        z_feat = self.z_fc(z).view(b, 128, 1, 1).expand(b, 128, h, w)

        hcat = torch.cat([x1_feat, z_feat], dim=1)
        x2_hat = self.decoder(hcat)
        return x2_hat
