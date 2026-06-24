import torch
import torch.nn.functional as F


class LossFactory(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.mse_loss = torch.nn.MSELoss() if self.cfg.train.loss.mse_loss else None
        self.fft_loss = FFTLoss() if self.cfg.train.loss.fft_loss else None

    def forward(self, pred, target, skip=None):
        device = pred.device
        dtype = pred.dtype
        if self.mse_loss:
            mse_loss = self.mse_loss(pred, target) * self.cfg.train.loss.mse_weight
        else:
            mse_loss = torch.tensor(0.0, device=device, dtype=dtype).detach()

        if self.fft_loss:
            if skip is not None:
                fft_loss = self.fft_loss(pred, skip) * self.cfg.train.loss.fft_weight
            else:
                fft_loss = self.fft_loss(pred, target) * self.cfg.train.loss.fft_weight
        else:
            fft_loss = torch.tensor(0.0, device=device, dtype=dtype).detach()

        return mse_loss, fft_loss


class FFTLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.criterion = torch.nn.L1Loss(reduction="none")

    def forward(self, pred, target):
        assert pred.shape == target.shape, f"{pred.shape=}, {target.shape=}"

        B, C, H, W = pred.shape
        dtype = pred.dtype
        device = pred.device
        pred = pred.to(torch.float32) + 1.0
        target = target.to(torch.float32) + 1.0

        dist = self.get_dist(H, W).repeat(B, C, 1, 1)
        dist = dist.to(device=device, dtype=dtype)

        freq_pred = torch.fft.fft2(pred, dim=(-1, -2), norm="ortho")
        freq_pred = torch.fft.fftshift(freq_pred, dim=(-1, -2))
        energy_pred = torch.abs(freq_pred) ** 2
        energy_pred = torch.log1p(energy_pred)

        freq_target = torch.fft.fft2(target, dim=(-1, -2), norm="ortho")
        freq_target = torch.fft.fftshift(freq_target, dim=(-1, -2))
        energy_target = torch.abs(freq_target) ** 2
        energy_target = torch.log1p(energy_target)

        loss = self.criterion(energy_pred, energy_target)
        loss *= dist

        return loss.mean()

    @torch.no_grad()
    def get_dist(self, h, w):
        x = torch.arange(h) - h / 2 + 0.5
        y = torch.arange(w) - w / 2 + 0.5
        x, y = torch.meshgrid(x, y, indexing="ij")
        x = torch.abs(x)
        y = torch.abs(y)
        xmax = torch.max(x)
        ymax = torch.max(y)
        x = x / xmax
        y = y / ymax
        x = torch.sin(x * (torch.pi / 2)) * x
        y = torch.sin(y * (torch.pi / 2)) * y

        dist = torch.maximum(x, y)

        return dist


class PerceptualLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def hinge_d_loss(self, logits_real, logits_fake):
        loss_real = torch.mean(F.relu(1.0 - logits_real))
        loss_fake = torch.mean(F.relu(1.0 + logits_fake))
        d_loss = 0.5 * (loss_real + loss_fake)
        return d_loss

    def forward(self, logits_fake, logits_real):
        return self.hinge_d_loss(logits_real, logits_fake)
