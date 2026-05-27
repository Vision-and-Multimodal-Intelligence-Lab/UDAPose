import json
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.utils.data
from crowdposetools.coco import COCO
from numpy.core.defchararray import array
from PIL import Image
from torchvision.transforms.v2 import functional as F

import datasets.transforms_exlpose as T
from datasets.data_util import preparing_dataset
from util.box_ops import box_cxcywh_to_xyxy, box_iou

from .ella_aug import ELLA

__all__ = ["build"]


class CocoDetection(torch.utils.data.Dataset):
    def __init__(self, root_path, image_set, transforms, return_masks, stage, use_ella=False):
        super(CocoDetection, self).__init__()
        if image_set == "train":
            assert stage in ["wl", "ll"], f"{stage=}, {image_set=}"
            print(f"image_set: {image_set}, stage: {stage}")
        self.stage = stage
        self.use_ella = use_ella
        if use_ella:
            if image_set != "train":
                self.use_ella = False
            else:
                print("================ Enable ELLA augmentation!!! ================")
                self.ella = ELLA()
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)
        self.image_set = image_set
        if image_set == "train":
            with open("data/mapping_list.json", "r", encoding="utf-8") as f:
                self.style_mapping_list = json.load(f)
            self.img_folder = root_path
            self.coco = COCO(root_path / "Annotations" / "ExLPose_train_WL.json")
            self.all_imgIds = []

            for coco in [self.coco]:
                imgIds = sorted(coco.getImgIds())
                for image_id in imgIds:
                    if coco.getAnnIds(imgIds=image_id) == []:
                        continue
                    ann_ids = coco.getAnnIds(imgIds=image_id)
                    target = coco.loadAnns(ann_ids)
                    num_keypoints = [obj["num_keypoints"] for obj in target]
                    if sum(num_keypoints) == 0:
                        continue
                    self.all_imgIds.append(image_id)
        else:
            self.img_folder = root_path
            # TODO: Extend to more test sets
            if self.stage == "wl":
                self.coco = COCO(root_path / "Annotations" / "ExLPose_test_WL.json")
            elif self.stage == "ll":
                self.coco = COCO(root_path / "Annotations" / "ExLPose_test_LL-A.json")
            elif self.stage == "lln":
                self.coco = COCO(root_path / "Annotations" / "ExLPose_test_LL-N.json")
            elif self.stage == "llh":
                self.coco = COCO(root_path / "Annotations" / "ExLPose_test_LL-H.json")
            elif self.stage == "lle":
                self.coco = COCO(root_path / "Annotations" / "ExLPose_test_LL-E.json")
            elif self.stage == "a7m3":
                self.coco = COCO(root_path / "Annotations" / "ExLPose-OC_test_A7M3.json")
            elif self.stage == "ricoh3":
                self.coco = COCO(root_path / "Annotations" / "ExLPose-OC_test_RICOH3.json")
            else:
                raise ValueError(f"Invalid stage: {self.stage}")
            imgIds = sorted(self.coco.getImgIds())
            self.all_imgIds = []
            for image_id in imgIds:
                self.all_imgIds.append(image_id)

    def __len__(self):
        return len(self.all_imgIds)

    def __getitem__(self, idx):
        img_light_level = None
        coco = self.coco
        if self.image_set != "train":
            return self.__getitem_eval__(idx)
        elif self.stage == "wl":
            img_light_level = "WL"
        else:
            rand = random.uniform(0.0, 1.0)
            if rand < 0.2:
                img_light_level = "WL"
            else:
                img_light_level = "LL"

        image_id = self.all_imgIds[idx]
        ann_ids = coco.getAnnIds(imgIds=image_id)
        target = coco.loadAnns(ann_ids)
        h, w = coco.loadImgs(image_id)[0]["height"], coco.loadImgs(image_id)[0]["width"]

        target = {"image_id": image_id, "annotations": target, "img_light_level": img_light_level}
        if self.image_set == "train" and img_light_level == "LL" and not self.use_ella:
            path = "data/synthetic"
            ver = random.choice(["normal", "hard", "extreme", "a7m3", "ricoh3"])
            # ver = random.choice(["normal", "hard", "extreme"])
            version_ids = self.style_mapping_list[f"{image_id}"][ver]
            version_id = random.choice(version_ids[:2])
            img = Image.open(os.path.join(path, f"{image_id}", f"stylized_{ver}_{version_id}.webp"))
            img = img.resize((w, h), Image.Resampling.NEAREST)
        else:
            img = Image.open(self.img_folder / coco.loadImgs(image_id)[0]["file_name"])
        img, target = self.prepare(img, target)
        if self.use_ella:
            img = self.ella.aug(F.to_image(img))
            img = F.to_pil_image(img)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target

    def __getitem_eval__(self, idx):
        coco = self.coco
        img_light_level = "LL"

        image_id = self.all_imgIds[idx]
        ann_ids = coco.getAnnIds(imgIds=image_id)
        target = coco.loadAnns(ann_ids)

        h, w = coco.loadImgs(image_id)[0]["height"], coco.loadImgs(image_id)[0]["width"]

        target = {"image_id": image_id, "annotations": target, "img_light_level": img_light_level}

        img = Image.open(self.img_folder / coco.loadImgs(image_id)[0]["file_name"])
        if img.size != (w, h):
            img = img.resize((w, h), Image.Resampling.NEAREST)
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        img_array = np.array(image)
        if len(img_array.shape) == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            image = Image.fromarray(img_array)
        image_id = target["image_id"]
        image_id = torch.tensor([image_id])
        anno = target["annotations"]
        anno = [obj for obj in anno if "iscrowd" not in obj or obj["iscrowd"] == 0]
        anno = [obj for obj in anno if obj["num_keypoints"] != 0]
        keypoints = [obj["keypoints"] for obj in anno]
        boxes = [obj["bbox"] for obj in anno]
        keypoints = torch.as_tensor(keypoints, dtype=torch.float32).reshape(-1, 14, 3)
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]  # (x, y, w, h) -> (x1, y1, x2, y2)
        # ?: Not sure really need clamp here for token
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)
        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)
        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        keypoints = keypoints[keep]
        if self.return_masks:
            masks = masks[keep]
        img_light_level = target["img_light_level"]
        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["iscrowd"] = iscrowd[keep]
        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])
        target["img_light_level"] = img_light_level
        return image, target


def make_coco_transforms(image_set, fix_size=False, args=None):
    normalize = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(
                [0.3457, 0.3460, 0.3463],
                [0.1477, 0.1482, 0.1483],
            ),
        ]
    )

    # config the params for data aug
    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    max_size = 1333
    scales2_resize = [400, 500, 600]
    scales2_crop = [384, 600]

    # update args from config files
    scales = getattr(args, "data_aug_scales", scales)
    max_size = getattr(args, "data_aug_max_size", max_size)
    scales2_resize = getattr(args, "data_aug_scales2_resize", scales2_resize)
    scales2_crop = getattr(args, "data_aug_scales2_crop", scales2_crop)

    # resize them
    data_aug_scale_overlap = getattr(args, "data_aug_scale_overlap", None)
    if data_aug_scale_overlap is not None and data_aug_scale_overlap > 0:
        data_aug_scale_overlap = float(data_aug_scale_overlap)
        scales = [int(i * data_aug_scale_overlap) for i in scales]
        max_size = int(max_size * data_aug_scale_overlap)
        scales2_resize = [int(i * data_aug_scale_overlap) for i in scales2_resize]
        scales2_crop = [int(i * data_aug_scale_overlap) for i in scales2_crop]

    datadict_for_print = {
        "scales": scales,
        "max_size": max_size,
        "scales2_resize": scales2_resize,
        "scales2_crop": scales2_crop,
    }
    print("data_aug_params:", json.dumps(datadict_for_print, indent=2))

    if image_set == "train":
        if fix_size:
            return T.Compose(
                [
                    T.RandomHorizontalFlip(),
                    T.RandomResize([(max_size, max(scales))]),
                    normalize,
                ]
            )

        return T.Compose(
            [
                T.RandomHorizontalFlip(),
                T.RandomSelect(
                    T.RandomResize(scales, max_size=max_size),
                    T.Compose(
                        [
                            T.RandomResize(scales2_resize),
                            T.RandomSizeCrop(*scales2_crop),
                            T.RandomResize(scales, max_size=max_size),
                        ]
                    ),
                ),
                normalize,
            ]
        )
    if image_set in ["val", "test"]:
        return T.Compose(
            [
                T.RandomResize([max(scales)], max_size=max_size),
                normalize,
            ]
        )

    raise ValueError(f"unknown {image_set}")


def build(image_set, args):
    root = Path(args.exlpose_path)
    dataset = CocoDetection(
        root,
        image_set,
        transforms=make_coco_transforms(image_set),
        return_masks=args.masks,
        stage=args.stage,
        use_ella=args.use_ella,
    )
    return dataset
