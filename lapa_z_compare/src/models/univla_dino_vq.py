import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Dinov2Model, T5EncoderModel, T5Tokenizer
from .base_model import BaseLatentModel
from .lapa_nsvq import StandardVQQuantizer

class UniVLADinoVQModel(BaseLatentModel):
    """
    UniVLA-style Task-centric DINO/VQ Latent Action Model.
    Reconstructs future DINOv2 features rather than raw pixels.
    Conditioned on T5 embeddings of task instructions (label text).
    """
    def __init__(self, latent_dim=32, num_codes=64, beta=0.25):
        super().__init__()
        
        # Load pre-trained frozen backbones
        print("Loading DINOv2 Vit-B/14...")
        self.dinov2 = Dinov2Model.from_pretrained("facebook/dinov2-base")
        for p in self.dinov2.parameters():
            p.requires_grad = False
            
        print("Loading T5 Encoder...")
        self.t5_tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-small")
        self.t5_model = T5EncoderModel.from_pretrained("google-t5/t5-small")
        for p in self.t5_model.parameters():
            p.requires_grad = False

        # Visual feature projection (from 2 * DINO_dim = 1536 to 256)
        self.visual_encoder = nn.Sequential(
            nn.Conv2d(1536, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Text embedding projection (from T5_dim = 512 to 256)
        self.text_proj = nn.Linear(512, 256)
        
        # Latent projection (visual 256 + text 256 = 512 -> latent_dim)
        self.z_fc = nn.Linear(512, latent_dim)
        
        # Quantizer
        self.quantizer = StandardVQQuantizer(num_codes=num_codes, latent_dim=latent_dim, beta=beta)
        
        # Decoder
        self.decoder_z_proj = nn.Linear(latent_dim, 128)
        self.decoder_text_proj = nn.Linear(512, 128)
        
        # Decodes combined feature maps [128 (z) + 128 (text) = 256] -> 768 (x_t+H)
        self.feature_decoder = nn.Sequential(
            nn.Conv2d(256, 768, 3, padding=1),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),
            nn.Conv2d(768, 768, 3, padding=1)
        )

    def get_dino_features(self, x):
        # DINOv2 expects size multiple of 14, standard is 224x224.
        if x.shape[-2:] != (224, 224):
            x = F.interpolate(x, size=(224, 224), mode="bicubic", align_corners=False)
            
        with torch.no_grad():
            outputs = self.dinov2(x)
            # last_hidden_state shape: [B, 257, 768]
            # Discard cls token and reshape patch tokens to [B, 768, 16, 16]
            patches = outputs.last_hidden_state[:, 1:, :]  # [B, 256, 768]
            patches = patches.permute(0, 2, 1).view(-1, 768, 16, 16)
        return patches

    def get_text_embedding(self, labels, device):
        if labels is None:
            # Fallback if no labels are provided
            return torch.zeros((1, 512), device=device)
            
        with torch.no_grad():
            inputs = self.t5_tokenizer(list(labels), return_tensors="pt", padding=True, truncation=True).to(device)
            outputs = self.t5_model(**inputs)
            # Mean pool over sequence dimension
            emb = outputs.last_hidden_state.mean(dim=1)  # [B, 512]
        return emb

    def encode(self, x_t, x_tpH, label=None):
        feat_t = self.get_dino_features(x_t)
        feat_tpH = self.get_dino_features(x_tpH)
        
        # Concatenate frame features along channel dimension
        feat_cat = torch.cat([feat_t, feat_tpH], dim=1) # [B, 1536, 16, 16]
        v_feat = self.visual_encoder(feat_cat).flatten(1) # [B, 256]
        
        t_emb = self.get_text_embedding(label, x_t.device)
        t_feat = F.relu(self.text_proj(t_emb)) # [B, 256]
        
        combined = torch.cat([v_feat, t_feat], dim=1) # [B, 512]
        z_e = self.z_fc(combined)
        
        z_out, _, _ = self.quantizer(z_e)
        return z_out

    def decode(self, x_t, z, label=None):
        feat_t = self.get_dino_features(x_t)
        b, _, h, w = feat_t.shape
        
        z_feat = self.decoder_z_proj(z).view(b, 128, 1, 1).expand(b, 128, h, w)
        
        t_emb = self.get_text_embedding(label, x_t.device)
        t_feat = self.decoder_text_proj(t_emb).view(b, 128, 1, 1).expand(b, 128, h, w)
        
        hcat = torch.cat([z_feat, t_feat], dim=1) # [B, 256, h, w]
        feat_tpH_hat = self.feature_decoder(hcat)
        return feat_tpH_hat

    def forward(self, x_t, x_tpH, label=None, structural=None):
        feat_t = self.get_dino_features(x_t)
        feat_tpH = self.get_dino_features(x_tpH)
        
        # Visual encoding
        feat_cat = torch.cat([feat_t, feat_tpH], dim=1)
        v_feat = self.visual_encoder(feat_cat).flatten(1)
        
        # Text encoding
        t_emb = self.get_text_embedding(label, x_t.device)
        t_feat = F.relu(self.text_proj(t_emb))
        
        # Combine
        combined = torch.cat([v_feat, t_feat], dim=1)
        z_e = self.z_fc(combined)
        
        # Quantize
        z_out, indices, vq_loss = self.quantizer(z_e)
        
        # Decode features
        b, _, h, w = feat_t.shape
        z_feat = self.decoder_z_proj(z_out).view(b, 128, 1, 1).expand(b, 128, h, w)
        t_decoder_feat = self.decoder_text_proj(t_emb).view(b, 128, 1, 1).expand(b, 128, h, w)
        
        hcat = torch.cat([z_feat, t_decoder_feat], dim=1)
        feat_tpH_hat = self.feature_decoder(hcat)
        
        # Reconstruction loss is computed in feature space
        rec_loss = F.mse_loss(feat_tpH_hat, feat_tpH)
        total_loss = rec_loss + vq_loss

        return {
            "z": z_out,
            "x_hat": feat_tpH_hat,
            "indices": indices,
            "loss_dict": {
                "loss": total_loss,
                "rec_loss": rec_loss,
                "vq_loss": vq_loss
            }
        }
