import torch


def rescale_stats(
    img: torch.Tensor,
    tgt_mean: torch.Tensor | float,
    tgt_std: torch.Tensor | float | None = None,
):
    ndim = img.ndim
    dtype = img.dtype
    device = img.device

    if ndim == 4:
        img = img.squeeze(0)

    img = img.type(torch.float64)

    if isinstance(tgt_mean, float):
        if tgt_std is not None:
            img_mean = img.mean((-1, -2, -3), keepdim=True)
            img_std = img.std((-1, -2, -3), keepdim=True)
            img = (img - img_mean) / (img_std + 1e-4) * tgt_std + tgt_mean
        else:
            img = img * (tgt_mean / (img.mean((-1, -2, -3), keepdim=True) + 1e-4))

    else:
        assert tgt_mean.ndim == 1 and tgt_mean.shape == (3,), f"{tgt_mean.shape=}"
        tgt_mean = tgt_mean[:, None, None]
        tgt_mean = tgt_mean.to(dtype=torch.float64, device=device)

        if tgt_std is not None:
            assert tgt_std.ndim == 1 and tgt_std.shape == (3,), f"{tgt_std.shape=}"
            tgt_std = tgt_std[:, None, None]
            tgt_std = tgt_std.to(dtype=torch.float64, device=device)

            img_mean = img.mean((-1, -2), keepdim=True)
            img_std = img.std((-1, -2), keepdim=True)
            img = (img - img_mean) / (img_std + 1e-4) * tgt_std + tgt_mean
        else:
            img = img * (tgt_mean / img.mean((-1, -2), keepdim=True) + 1e-4)

    assert not img.isinf().any(), f"{img.isinf().any()=}"
    assert not img.isnan().any(), f"{img.isnan().any()=}"

    if ndim == 4:
        img = img.unsqueeze(0)

    return img.round().clip(0, 255).type(dtype)


# def rescale_stats_fixed_factor(
#     img: torch.Tensor,
# ):
#     # !: ablation study
#     ndim = img.ndim
#     dtype = img.dtype
#     device = img.device
#     fixed_factor = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=torch.float64) * 255
#     fixed_factor = fixed_factor[:, None, None]
#     fixed_std = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=torch.float64) * 255
#     fixed_std = fixed_std[:, None, None]

#     if ndim == 4:
#         img = img.squeeze(0)

#     img = img.type(torch.float64)
#     rgb_mean = img.mean((-1, -2), keepdim=True)
#     rgb_std = img.std((-1, -2), keepdim=True)

#     img = (img - rgb_mean) / rgb_std * fixed_std + fixed_factor

#     assert not img.isinf().any(), f"{img.isinf().any()=}"
#     assert not img.isnan().any(), f"{img.isnan().any()=}"

#     if ndim == 4:
#         img = img.unsqueeze(0)

#     return img.round().clip(0, 255).type(dtype)
