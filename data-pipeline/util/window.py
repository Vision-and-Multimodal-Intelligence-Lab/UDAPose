import einops
import torch
from torchvision.utils import draw_bounding_boxes

H = 102
W = 162


@torch.inference_mode()
def crop_patch(
    img,
    window_h=H,
    window_w=W,
    mean_percentile=0.5,
    std_percentile=0.5,
    energy_map="rgb",
):
    assert img.ndim == 4, f"{img.shape=}"
    device = img.device

    img_fl = img.float()
    img_fl = img_fl / 255

    if energy_map == "rgb":
        img_fl = einops.reduce(img_fl, "b c h w -> b 1 h w", "mean")

    elif energy_map == "hsl":
        img_fl_max = einops.reduce(img_fl, "b c h w -> b 1 h w", "max")
        img_fl_min = einops.reduce(img_fl, "b c h w -> b 1 h w", "min")
        img_fl = (img_fl_max + img_fl_min) / 2

    else:
        raise ValueError(f"Invalid energy map: {energy_map}")

    # !: use avg_pool2d to boost
    kernel_size = (window_h, window_w)
    stride = 1
    window_mean = torch.nn.functional.avg_pool2d(img_fl, kernel_size, stride)
    window_sq_mean = torch.nn.functional.avg_pool2d(img_fl.pow(2), kernel_size, stride)
    # σ = √(E[x²] – μ²)
    window_std = (window_sq_mean - window_mean.pow(2)).clamp_min(0).sqrt()

    # find smallest mean_percentile indexes
    _, indexes_1d = torch.topk(
        window_mean.flatten(),
        round(window_mean.numel() * mean_percentile),
        largest=False,
    )

    indexes_4d = torch.unravel_index(indexes_1d, window_mean.shape)

    # now we have smallest mean_percentile indexes, we find the std_percentile in the mean_percentile
    selected_stds = window_std[indexes_4d]
    _, sorted_indexes = torch.sort(selected_stds.flatten(), stable=True)
    pos = sorted_indexes[round(selected_stds.numel() * std_percentile)]

    pos_h = indexes_4d[2][pos]
    pos_w = indexes_4d[3][pos]

    crop_out = img[..., pos_h : pos_h + window_h, pos_w : pos_w + window_w].detach()

    crop_out = crop_out.repeat(1, 1, img.shape[2] // window_h, img.shape[3] // window_w)

    return crop_out, (pos_h, pos_w)


# H = 204
# W = 324


# @apply_forward_hook
# @torch.inference_mode()
# def crop_patch(
#     content_img,
#     style_img,
#     window_h=H,
#     window_w=W,
#     energy_map="rgb",
# ):
#     assert content_img.shape[0] == 1 and style_img.shape[0] == 1, (
#         "Batch size for both content and style images must be 1"
#     )
#     assert content_img.ndim == style_img.ndim == 4, (
#         f"{content_img.shape=}, {style_img.shape=}"
#     )
#     device = content_img.device
#     nbins = 256

#     # Remove batch dimension
#     content_img = content_img.squeeze(0)
#     style_img_squeezed = style_img.squeeze(0)

#     # Process content image to get the target histogram
#     content_fl = content_img.float() / 255
#     if energy_map == "rgb":
#         content_fl = einops.reduce(content_fl, "c h w -> 1 h w", "mean")
#     elif energy_map == "hsl":
#         img_fl_max = einops.reduce(content_fl, "c h w -> 1 h w", "max")
#         img_fl_min = einops.reduce(content_fl, "c h w -> 1 h w", "min")
#         content_fl = (img_fl_max + img_fl_min) / 2
#     else:
#         raise ValueError(f"Invalid energy map: {energy_map}")

#     # Batched histogram for content image
#     content_flat = content_fl.flatten()
#     bins = (content_flat * (nbins - 1)).long()
#     hist_whole = torch.bincount(bins, minlength=nbins).float()
#     hist_whole = hist_whole / hist_whole.sum()

#     # Process style image
#     style_fl = style_img_squeezed.float() / 255
#     if energy_map == "rgb":
#         style_fl = einops.reduce(style_fl, "c h w -> 1 h w", "mean")
#     elif energy_map == "hsl":
#         img_fl_max = einops.reduce(style_fl, "c h w -> 1 h w", "max")
#         img_fl_min = einops.reduce(style_fl, "c h w -> 1 h w", "min")
#         style_fl = (img_fl_max + img_fl_min) / 2
#     else:
#         raise ValueError(f"Invalid energy map: {energy_map}")

#     # To avoid OOM, we process the image row by row of patches.
#     H_out = style_img_squeezed.shape[-2] - window_h + 1
#     W_out = style_img_squeezed.shape[-1] - window_w + 1
#     jsd_matrix = torch.zeros((H_out, W_out), device=device)

#     for h_idx in range(H_out):
#         # Extract one row of patches
#         row_slice = style_fl[
#             :, h_idx : h_idx + window_h, :
#         ]  # shape (1, window_h, W_img)
#         # unfold expects a 4D tensor, (N, C, H, W)
#         row_windows = F.unfold(
#             row_slice.unsqueeze(0), kernel_size=(window_h, window_w)
#         )  # (1, C*kh*kw, W_out)
#         row_windows = row_windows.squeeze(0).permute(1, 0)  # (W_out, C*kh*kw)

#         # Batched histogram for the row of windows
#         bins = (row_windows * (nbins - 1)).long()
#         hist_windows_row = torch.zeros(
#             (bins.shape[0], nbins), device=device, dtype=torch.float
#         )
#         ones = torch.ones_like(bins, dtype=torch.float)
#         hist_windows_row.scatter_add_(1, bins, ones)
#         hist_windows_row = hist_windows_row / hist_windows_row.sum(dim=1, keepdim=True)

#         # Batched Jensen-Shannon divergence for the row
#         m = 0.5 * (hist_windows_row + hist_whole)
#         eps = 1e-8
#         kl_p_m = F.kl_div((m + eps).log(), hist_windows_row, reduction="none").sum(-1)
#         kl_q_m = F.kl_div(
#             (m + eps).log(),
#             hist_whole.expand_as(hist_windows_row),
#             reduction="none",
#         ).sum(-1)
#         jsd_scores_row = 0.5 * (kl_p_m + kl_q_m)

#         jsd_matrix[h_idx, :] = jsd_scores_row

#     _, min_idx = torch.min(jsd_matrix.flatten(), dim=0)
#     pos_h, pos_w = torch.unravel_index(min_idx, jsd_matrix.shape)

#     crop_out = style_img[
#         ..., pos_h : pos_h + window_h, pos_w : pos_w + window_w
#     ].detach()  # (1, 3, H, W)

#     # Repeat the cropped patch
#     crop_out = crop_out.repeat(
#         1,
#         1,
#         style_img_squeezed.shape[1] // window_h,
#         style_img_squeezed.shape[2] // window_w,
#     )

#     return crop_out, (pos_h, pos_w)


def reflect_extend(input: torch.Tensor, h_times: int, w_times: int):
    assert input.ndim == 4, f"{input.shape=}"
    device = input.device

    b, c, h, w = input.shape

    orig = input.clone()
    h_reversed = torch.flip(input, dims=(2,))
    for i in range(1, h_times):
        if i % 2 == 0:
            input = torch.cat([input, orig], dim=2)
        else:
            input = torch.cat([input, h_reversed], dim=2)

    orig = input.clone()
    w_reversed = torch.flip(input, dims=(3,))
    for i in range(1, w_times):
        if i % 2 == 0:
            input = torch.cat([input, orig], dim=3)
        else:
            input = torch.cat([input, w_reversed], dim=3)

    return input


def draw_patch(
    img: torch.Tensor, pos_h: int, pos_w: int, window_h=H, window_w=W
) -> torch.Tensor:
    assert img.ndim == 4, f"{img.shape=}"
    device = img.device

    img = img[0]
    img = draw_bounding_boxes(
        img,
        torch.tensor(
            [[pos_w, pos_h, pos_w + window_w, pos_h + window_h]], device=device
        ),
        colors="red",
        width=2,
    ).to(device)

    return img[None]
