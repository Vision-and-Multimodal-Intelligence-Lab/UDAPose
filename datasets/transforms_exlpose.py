# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Transforms and data augmentation for both image + bbox.
"""

import os
import random

import PIL
import torch
import torchvision
import torchvision.transforms as T
from torchvision import io
import torchvision.transforms.functional as F
import einops
from util.box_ops import box_xyxy_to_cxcywh
from util.misc import interpolate

CROWDPOSE_CONNECTIVITY = [
    [12, 13],  # head -> neck
    [13, 0],  # neck -> l_shoulder
    [13, 1],  # neck -> r_shoulder
    [0, 2],  # l_shoulder -> l_elbow
    [2, 4],  # l_elbow -> l_wrist
    [1, 3],  # r_shoulder -> r_elbow
    [3, 5],  # r_elbow -> r_wrist
    [13, 7],  # neck -> r_hip
    [13, 6],  # neck -> l_hip
    [7, 9],  # r_hip -> r_knee
    [9, 11],  # r_knee -> r_ankle
    [6, 8],  # l_hip -> l_knee
    [8, 10],  # l_knee -> l_ankle
]


def crop(image, target, region):
    cropped_image = F.crop(image, *region)

    target = target.copy()
    i, j, h, w = region

    # should we do something wrt the original size?
    target["size"] = torch.tensor([h, w])

    fields = ["labels", "area", "iscrowd", "keypoints"]

    if "boxes" in target:
        boxes = target["boxes"]
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
        target["boxes"] = cropped_boxes.reshape(-1, 4)
        target["area"] = area
        fields.append("boxes")

    if "masks" in target:
        # FIXME should we update the area here if there are no boxes?
        target["masks"] = target["masks"][:, i : i + h, j : j + w]
        fields.append("masks")

    # remove elements for which the boxes or masks that have zero area
    if "boxes" in target or "masks" in target:
        # favor boxes selection when defining which elements to keep
        # this is compatible with previous implementation
        if "boxes" in target:
            cropped_boxes = target["boxes"].reshape(-1, 2, 2)
            keep = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
        else:
            keep = target["masks"].flatten(1).any(1)

        for field in fields:
            target[field] = target[field][keep]

    if "keypoints" in target:
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        keypoints = target["keypoints"]
        cropped_keypoints = keypoints.view(-1, 3)[:, :2] - torch.as_tensor([j, i])
        cropped_keypoints = torch.min(cropped_keypoints, max_size)
        cropped_keypoints = cropped_keypoints.clamp(min=0)
        cropped_keypoints = torch.cat([cropped_keypoints, keypoints.view(-1, 3)[:, 2].unsqueeze(1)], dim=1)
        target["keypoints"] = cropped_keypoints.view(target["keypoints"].shape[0], 14, 3)
        # fields.append("keypoints")

    return cropped_image, target


def hflip(image, target):
    flipped_image = F.hflip(image)

    w, h = image.size

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
        target["boxes"] = boxes

    if "keypoints" in target:
        flip_pairs = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11]]
        keypoints = target["keypoints"]
        keypoints[:, :, 0] = w - keypoints[:, :, 0] - 1
        for pair in flip_pairs:
            keypoints[:, pair[0], :], keypoints[:, pair[1], :] = (
                keypoints[:, pair[1], :],
                keypoints[:, pair[0], :].clone(),
            )
        target["keypoints"] = keypoints

    if "masks" in target:
        target["masks"] = target["masks"].flip(-1)

    return flipped_image, target


def resize(image, target, size, max_size=None):
    # size can be min_size (scalar) or (w, h) tuple

    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(image.size, size, max_size)
    rescaled_image = F.resize(image, size)

    if target is None:
        return rescaled_image, None

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
    ratio_width, ratio_height = ratios

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["boxes"] = scaled_boxes

    if "area" in target:
        area = target["area"]
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area

    if "keypoints" in target:
        keypoints = target["keypoints"]
        scaled_keypoints = keypoints * torch.as_tensor([ratio_width, ratio_height, 1])
        target["keypoints"] = scaled_keypoints

    h, w = size
    target["size"] = torch.tensor([h, w])

    if "masks" in target:
        target["masks"] = interpolate(target["masks"][:, None].float(), size, mode="nearest")[:, 0] > 0.5

    return rescaled_image, target


def pad(image, target, padding):
    # assumes that we only pad on the bottom right corners
    padded_image = F.pad(image, (0, 0, padding[0], padding[1]))
    if target is None:
        return padded_image, None
    target = target.copy()
    # should we do something wrt the original size?
    target["size"] = torch.tensor(padded_image.size[::-1])
    if "masks" in target:
        target["masks"] = torch.nn.functional.pad(target["masks"], (0, padding[0], 0, padding[1]))
    return padded_image, target


class ResizeDebug(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        return resize(img, target, self.size)


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        region = T.RandomCrop.get_params(img, self.size)
        return crop(img, target, region)


class RandomSizeCrop(object):
    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img: PIL.Image.Image, target: dict):
        w = random.randint(self.min_size, min(img.width, self.max_size))
        h = random.randint(self.min_size, min(img.height, self.max_size))
        region = T.RandomCrop.get_params(img, [h, w])
        return crop(img, target, region)


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        image_width, image_height = img.size
        crop_height, crop_width = self.size
        crop_top = int(round((image_height - crop_height) / 2.0))
        crop_left = int(round((image_width - crop_width) / 2.0))
        return crop(img, target, (crop_top, crop_left, crop_height, crop_width))


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return hflip(img, target)
        return img, target


class RandomResize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, target=None):
        size = random.choice(self.sizes)
        return resize(img, target, size, self.max_size)


class RandomPad(object):
    def __init__(self, max_pad):
        self.max_pad = max_pad

    def __call__(self, img, target):
        pad_x = random.randint(0, self.max_pad)
        pad_y = random.randint(0, self.max_pad)
        return pad(img, target, (pad_x, pad_y))


class RandomSelect(object):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """

    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return self.transforms1(img, target)
        return self.transforms2(img, target)


class ToTensor(object):
    def __call__(self, img, target):
        return F.to_tensor(img), target


class RandomErasing(object):
    def __init__(self, *args, **kwargs):
        self.eraser = T.RandomErasing(*args, **kwargs)

    def __call__(self, img, target):
        return self.eraser(img), target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        img_light_level = target["img_light_level"]

        if img_light_level == "WL":
            target["unnormalized_image"] = image.clone()
        elif img_light_level == "LL":
            assert image.is_floating_point(), "image is not floating point"
            rgb_mean = image.mean()
            vis_image = image * (0.4 / (rgb_mean + 1e-4))
            vis_image = vis_image.clone().clip(0, 1)
            target["unnormalized_image"] = (vis_image * 255.0).round().to(torch.uint8)
        else:
            raise ValueError(f"Invalid img_light_level: {img_light_level}")
        image = F.normalize(image, mean=self.mean, std=self.std)

        assert not image.isinf().any(), "image is inf"
        assert not image.isnan().any(), "image is nan"

        if target is None:
            return image, None
        target = target.copy()
        h, w = image.shape[-2:]
        if "boxes" in target:
            boxes = target["boxes"]
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["boxes"] = boxes
        if "area" in target:
            area = target["area"]
            area = area / (torch.tensor(w, dtype=torch.float32) * torch.tensor(h, dtype=torch.float32))
            target["area"] = area
        else:
            area = target["boxes"][:, -1] * target["boxes"][:, -2]
            target["area"] = area
        if "keypoints" in target:
            keypoints = target["keypoints"]  # (4, 14, 3) (num_person, num_keypoints, 3)
            V = keypoints[:, :, 2]  # visibility of the keypoints torch.Size([number of persons, 14])
            V[V == 2] = 1
            Z = keypoints[:, :, :2]
            Z = Z.contiguous().view(-1, 2 * 14)
            Z = Z / torch.tensor([w, h] * 14, dtype=torch.float32)
            all_keypoints = torch.cat([Z, V], dim=1)  # torch.Size([number of persons, 28+14])
            target["keypoints"] = all_keypoints  # (num_person, 28+14) -> (xyxyxyzzz)

            # Propogate relative position within the virtualbbox (1.12 * bbox)
            reshaped_Z = Z.view(-1, 14, 2).detach()  # (num_person, 14, 2)
            gt_reshape = reshaped_Z.clone()
            cx, cy, box_w, box_h = boxes.unbind(-1)  # (num_person,)
            box_w, box_h = box_w * 1.12, box_h * 1.12
            tl = torch.stack([cx - box_w / 2, cy - box_h / 2], dim=-1)  # (num_person, 2)
            br = torch.stack([cx + box_w / 2, cy + box_h / 2], dim=-1)  # (num_person, 2)
            tr = torch.stack([cx + box_w / 2, cy - box_h / 2], dim=-1)  # (num_person, 2)
            bl = torch.stack([cx - box_w / 2, cy + box_h / 2], dim=-1)  # (num_person, 2)

            zeros = torch.zeros_like(cx)  # (num_person,)
            ones = torch.ones_like(cx) * 256.0  # (num_person,)

            vae_tl = torch.stack([zeros, zeros], dim=-1)  # (num_person, 2)
            vae_br = torch.stack([ones, ones], dim=-1)  # (num_person, 2)
            vae_tr = torch.stack([ones, zeros], dim=-1)  # (num_person, 2)
            vae_bl = torch.stack([zeros, ones], dim=-1)  # (num_person, 2)

            src_pts = torch.stack([tl, br, tr, bl], dim=0)  # (4, num_person, 2)
            src_pts = src_pts.transpose(0, 1).contiguous()  # (num_person, 4, 2)

            dst_pts = torch.stack([vae_tl, vae_br, vae_tr, vae_bl], dim=0)  # (4, num_person, 2)
            dst_pts = dst_pts.transpose(0, 1).contiguous()  # (num_person, 4, 2)

            M = get_affine_transform(src_pts, dst_pts)  # (num_person, 2, 3)
            M_inv = get_affine_transform(dst_pts, src_pts)
            reshaped_Z = apply_affine_transform(reshaped_Z, M)  # (num_person, 14, 2)
            keypoints_vis = apply_affine_transform(reshaped_Z, M_inv)  # (num_person, 14, 2)
            # --------------------------------------------------------------------------------
            # Sanity check: the forward + inverse transforms should be an identity mapping.
            # Validate that the recovered coordinates match the original ground-truth ones.
            if not torch.allclose(keypoints_vis, gt_reshape, atol=1e-4):
                max_diff = (keypoints_vis - gt_reshape).abs().max()
                raise RuntimeError(f"Affine transform consistency check failed. Max deviation: {max_diff.item():.6f}")
            # --------------------------------------------------------------------------------
            reshaped_Z = torch.cat([reshaped_Z, V[..., None]], dim=-1)  # (num_person, 14, 3)

            target["vqvae_keypoints"] = reshaped_Z

        return image, target


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


def generate_target(joints, heatmap_size, sigma):
    """
    :param joints:  [num_joints, 3]
    :return: target
    """
    num_keypoints, _ = joints.shape
    target_weight = torch.ones(num_keypoints, 1)
    target_weight[:, 0] = joints[:, -1]
    # target_type == 'gaussian'
    target = torch.zeros(num_keypoints, heatmap_size[1], heatmap_size[0])

    tmp_size = sigma * 3

    for joint_id in range(num_keypoints):
        target_weight[joint_id] = adjust_target_weight(
            joints[joint_id], heatmap_size, target_weight[joint_id], tmp_size
        )

        if target_weight[joint_id] == 0:
            continue

        mu_x = joints[joint_id][0]
        mu_y = joints[joint_id][1]

        # 生成过程与hrnet的heatmap size不一样
        x = torch.arange(0, heatmap_size[0], 1)
        y = torch.arange(0, heatmap_size[1], 1)
        y = y.unsqueeze(-1)

        v = target_weight[joint_id]
        if v > 0.5:
            target[joint_id] = torch.exp(-((x - mu_x) ** 2 + (y - mu_y) ** 2) / (2 * sigma**2))

    return target


def adjust_target_weight(joint, heatmap_size, target_weight, tmp_size):
    # feat_stride = self.image_size / self.heatmap_size
    mu_x = joint[0]
    mu_y = joint[1]
    # Check that any part of the gaussian is in-bounds
    ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
    br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
    if ul[0] >= heatmap_size[0] or ul[1] >= heatmap_size[1] or br[0] < 0 or br[1] < 0:
        # If not, just return the image as is
        target_weight = 0
    return target_weight


# -----------------------------------------------------------------------------
# Affine transform utils
# -----------------------------------------------------------------------------


def get_affine_transform(src_pts: torch.Tensor, dst_pts: torch.Tensor) -> torch.Tensor:
    """
    Args:
        src_pts: Source points, shape (N, 3, 2)
        dst_pts: Destination points, shape (N, 3, 2)
    Returns:
        Affine transform matrix, shape (N, 2, 3)
    """

    # Cast to float64 for better numerical stability
    src_pts64 = src_pts.to(torch.float64)
    dst_pts64 = dst_pts.to(torch.float64)

    N, P, _ = src_pts64.shape
    assert P >= 3, f"Need at least 3 point correspondences, got {P}"
    assert src_pts64.shape == dst_pts64.shape, f"{src_pts64.shape=} vs {dst_pts64.shape=}"

    # Build homogeneous coordinates (x, y, 1)
    ones = torch.ones((N, P, 1), dtype=torch.float64, device=src_pts64.device)
    A = torch.cat([src_pts64, ones], dim=-1)  # (N, P, 3)

    # Solve least-squares A @ X = dst  → X shape (N, 3, 2)
    X = torch.linalg.lstsq(A, dst_pts64).solution  # (N, 3, 2)

    # Rearrange to (N, 2, 3) for apply_affine_transform (which expects row-major 2×3)
    M = X.transpose(1, 2).contiguous()  # (N, 2, 3)

    return M.to(src_pts.dtype)


def apply_affine_transform(src_pts: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    """
    Args:
        src_pts: Source points, shape (N, D, 2)
        M: Affine transform matrix, shape (N, 2, 3)
    Returns:
        Transformed points, shape (N, D, 2)
    """
    assert src_pts.shape[0] == M.shape[0], f"{src_pts.shape=} and {M.shape=}"
    assert src_pts.shape[-1] == 2, f"{src_pts.shape[-1]=}"
    assert M.shape[1:] == (2, 3), f"{M.shape=}"

    assert src_pts.is_floating_point() and M.is_floating_point(), f"{src_pts.dtype=} and {M.dtype=}"

    N, D, _ = src_pts.shape
    device = src_pts.device

    # Promote to float64 for higher numerical stability
    src64 = src_pts.to(torch.float64)
    M64 = M.to(torch.float64)

    ones = torch.ones((N, D, 1), dtype=torch.float64, device=device)
    src64_h = torch.cat([src64, ones], dim=-1)  # (N, D, 3)
    dst64 = src64_h @ M64.transpose(1, 2)  # (N, D, 3)

    return dst64[:, :, :2].to(src_pts.dtype)
