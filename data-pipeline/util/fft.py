import torch
import einops

H_RATIO = W_RATIO = 0.16


def reconstruct_high_freq(img: torch.Tensor):
    """
    Reconstruct high-frequency components from the image.
    """
    device = img.device

    assert img.dtype == torch.uint8, f"{img.dtype=}"
    assert img.ndim == 4 and img.shape[1] == 3, f"{img.shape=}"

    img = img.float()

    # bright up
    rgb_mean = einops.reduce(img, "b c h w -> b 1 1 1", "mean")
    img = img * (255 * 0.449 / rgb_mean)
    img = img.round().clamp(0, 255)

    img = img / 255.0  # [0, 255] -> [0, 1]

    b, c, h, w = img.shape
    ori_img_mean = einops.reduce(img, "b c h w -> b c 1 1", "mean")

    crop_h = round(h * H_RATIO)
    crop_w = round(w * W_RATIO)

    mask = torch.zeros_like(img)
    mask[:, :, :crop_h, :] = 1.0
    mask[:, :, :, :crop_w] = 1.0
    mask[:, :, -crop_h:, :] = 1.0
    mask[:, :, :, -crop_w:] = 1.0

    freq = torch.fft.fft2(img, dim=(-1, -2), norm="ortho")
    freq = torch.fft.fftshift(freq, dim=(-1, -2))

    freq = freq * mask

    freq = torch.fft.ifftshift(freq, dim=(-1, -2))
    recon = torch.fft.ifft2(freq, dim=(-1, -2), norm="ortho")
    recon = recon.real

    # shift the reconstruction to style mean
    recon_mean = einops.reduce(recon, "b c h w -> b c 1 1", "mean")
    recon = recon + (ori_img_mean - recon_mean)
    recon = recon * 255.0

    return recon.round().clamp(0, 255).to(torch.uint8)
