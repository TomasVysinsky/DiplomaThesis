import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum

class SolutionApproach(Enum):
    SLIDING_WINDOW_ONE_RESULT = 1,
    SLIDING_WINDOW_MULTIPLE_RESULTS = 2,
    SLIDING_WINDOW_TWO_ARMS_MULTIPLE_RESULTS = 3,
    MULTIPLE_CATEGORIES = 4,
    MULTIPLE_CATEGORIES_SINGLE_RESULT = 5
    


class CnnBaseNetworkRadar(nn.Module):
    def __init__(self, in_channels : int = 4, num_classes: int = 3, seq_len: int = 50, dropout_rate: float = 0.3):
        super().__init__()


        # --- extrakcia čŕt z časového radu ---------------------------------
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),


            nn.Conv1d(16, 32, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )

        # --- globálne zhrnutie časového radu (pooling cez čas) -------------
        # spraví z (batch, 64, L) → (batch, 64, 1)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # --- klasifikačná hlavička: 3 logity pre 3 triedy ------------------
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(32, num_classes)
        )

    # ----------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extrakcia čŕt z časového radu
        x = self.features(x)            # (batch, 64, L)

        # Globálny pooling cez čas
        x = self.global_pool(x)         # (batch, 64, 1)
        x = x.squeeze(-1)               # (batch, 64)

        # Klasifikačná vrstva → logity
        logits = self.classifier(x)     # (batch, num_classes)

        return logits                   # použiješ BCEWithLogitsLoss

