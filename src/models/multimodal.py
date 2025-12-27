import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F


class MultiModalNet(nn.Module):
    """Simple multimodal dual-encoder model.

    - Two encoders (ResNet18) for CT and MRI (can share weights via `share_weights=True`).
    - Fusion by concatenation of encoder features.
    - Heads:
      - classification: num_classes (default 3)
      - stage regression: single output
      - survival risk: single output (regression/risk score)
    """

    def __init__(self, num_classes=3, share_weights=False, pretrained=False):
        super().__init__()
        # CT encoder
        self.ct_encoder = models.resnet18(pretrained=pretrained)
        self.ct_encoder.fc = nn.Identity()

        # MRI encoder
        if share_weights:
            self.mri_encoder = self.ct_encoder
        else:
            self.mri_encoder = models.resnet18(pretrained=pretrained)
            self.mri_encoder.fc = nn.Identity()

        # feature dim from resnet18 final (512)
        feat_dim = 512

        # fusion dim = sum of available encoders
        self.fusion_dim = feat_dim * 2

        # classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

        # stage regression head
        self.stage_head = nn.Sequential(
            nn.Linear(self.fusion_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # survival risk head
        self.surv_head = nn.Sequential(
            nn.Linear(self.fusion_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, ct=None, mri=None):
        # ct, mri: tensors or None. Expect shape [B, C, H, W]
        device = next(self.parameters()).device

        features = []
        if ct is not None:
            ct = ct.to(device)
            ct_f = self.ct_encoder(ct)  # [B, feat_dim]
            features.append(ct_f)
        if mri is not None:
            mri = mri.to(device)
            mri_f = self.mri_encoder(mri)
            features.append(mri_f)

        if not features:
            # If no modality is provided, this should not happen
            # Return a dummy output to avoid crash
            raise ValueError("At least one modality (CT or MRI) must be provided")
        elif len(features) == 1:
            # Only one modality present - use it directly, pad to fusion_dim
            fused = features[0]
            if fused.size(1) != self.fusion_dim:
                pad = self.fusion_dim - fused.size(1)
                batch_size = fused.size(0)
                zeros = torch.zeros((batch_size, pad), device=device, dtype=fused.dtype)
                fused = torch.cat([fused, zeros], dim=1)
        else:
            # Both modalities present - concatenate
            fused = torch.cat(features, dim=1)

        # normalize fused features for stability
        fused = F.normalize(fused, dim=1)

        cls_logits = self.classifier(fused)
        stage_out = self.stage_head(fused).squeeze(1)
        surv_out = self.surv_head(fused).squeeze(1)

        return {
            'logits': cls_logits,
            'stage': stage_out,
            'survival': surv_out
        }
