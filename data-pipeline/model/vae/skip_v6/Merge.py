import torch
from torch import nn


# ==== Modify from QuadPrior ====
class Merge(nn.Module):
    def __init__(self, merged_channels, out_channels, hidden_channels=None):
        super().__init__()

        if hidden_channels is None:
            hidden_channels = out_channels * 2
        else:
            hidden_channels = hidden_channels

        self.convs = nn.Sequential(
            nn.Conv2d(merged_channels, hidden_channels * 2, kernel_size=5, padding=2),
            GEGLU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1),
        )

        nn.init.zeros_(self.convs[2].weight.data)
        nn.init.zeros_(self.convs[2].bias.data)

    def forward(self, main_h, skip_h):
        h = self.convs(torch.cat([main_h, skip_h], dim=1))

        return h


class GEGLU(nn.Module):
    def __init__(self):
        super().__init__()

        self.gelu = nn.GELU(approximate="tanh")

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * self.gelu(x2)
