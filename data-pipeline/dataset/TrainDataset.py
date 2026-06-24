"""
This is the dataloader to fintune VAE from Stable Diffusion.
In this file, we would use a combination of LL-N, LL-H, LL-E, A7M3, RICOH3.
To ensafe our claim, we would only use test set images.
"""

import os
import random

import albumentations as A
import cv2
import numpy as np
import torch
from crowdposetools.coco import COCO
from torch.utils.data import DataLoader, Dataset

from util.suppress_stdout import suppress_stdout

"""
LLN + LLH + LLE + A7M3 + RICOH3 = 120 + 120 + 120 + 180 + 180 = 720
"""


class TrainDataset(Dataset):
    def __init__(self, cfg):
        super().__init__()

        self.exlpose_root = cfg.data.exlpose_root

        assert os.path.exists(self.exlpose_root), f"{self.exlpose_root} does not exist."

        with suppress_stdout():
            self.normal_coco = COCO(
                os.path.join(self.exlpose_root, "Annotations", "ExLPose_test_LL-N.json")
            )
            self.hard_coco = COCO(
                os.path.join(self.exlpose_root, "Annotations", "ExLPose_test_LL-H.json")
            )
            self.extreme_coco = COCO(
                os.path.join(self.exlpose_root, "Annotations", "ExLPose_test_LL-E.json")
            )
            self.a7m3_coco = COCO(
                os.path.join(
                    self.exlpose_root, "Annotations", "ExLPose-OC_test_A7M3.json"
                )
            )
            self.ricoh3_coco = COCO(
                os.path.join(
                    self.exlpose_root, "Annotations", "ExLPose-OC_test_RICOH3.json"
                )
            )
        self.cocos = [
            self.normal_coco,
            self.hard_coco,
            self.extreme_coco,
            self.a7m3_coco,
            self.ricoh3_coco,
        ]

        self.normal_img_ids = sorted(self.normal_coco.getImgIds())
        self.hard_img_ids = sorted(self.hard_coco.getImgIds())
        self.extreme_img_ids = sorted(self.extreme_coco.getImgIds())
        self.a7m3_img_ids = sorted(self.a7m3_coco.getImgIds())
        self.ricoh3_img_ids = sorted(self.ricoh3_coco.getImgIds())

        self.normal_selected_img_ids = random.sample(self.normal_img_ids, 120)
        self.hard_selected_img_ids = random.sample(self.hard_img_ids, 120)
        self.extreme_selected_img_ids = random.sample(self.extreme_img_ids, 120)
        self.a7m3_selected_img_ids = random.sample(self.a7m3_img_ids, 180)
        self.ricoh3_selected_img_ids = random.sample(self.ricoh3_img_ids, 180)
        self.img_ids = [
            self.normal_selected_img_ids,
            self.hard_selected_img_ids,
            self.extreme_selected_img_ids,
            self.a7m3_selected_img_ids,
            self.ricoh3_selected_img_ids,
        ]

        self.normal_len = len(self.normal_selected_img_ids)
        self.hard_len = len(self.hard_selected_img_ids)
        self.extreme_len = len(self.extreme_selected_img_ids)
        self.a7m3_len = len(self.a7m3_selected_img_ids)
        self.ricoh3_len = len(self.ricoh3_selected_img_ids)
        self.total_len = (
            self.normal_len
            + self.hard_len
            + self.extreme_len
            + self.a7m3_len
            + self.ricoh3_len
        )

        self.transforms = A.Compose(
            [
                A.SmallestMaxSize(512),
                A.RandomCrop(320, 512),
                A.HorizontalFlip(),
                A.VerticalFlip(),
                A.ToTensorV2(),
            ]
        )

    def __len__(self):
        return self.total_len

    def get_correct_index(self, index):
        if index < self.normal_len:
            return index, 0
        index -= self.normal_len
        if index < self.hard_len:
            return index, 1
        index -= self.hard_len
        if index < self.extreme_len:
            return index, 2
        index -= self.extreme_len
        if index < self.a7m3_len:
            return index, 3
        index -= self.a7m3_len
        return index, 4

    def __getitem__(self, index):
        z0_img = self.get_img_from_index(index)

        skip_img = z0_img.clone()
        tgt_img = z0_img.clone()

        return z0_img, skip_img, tgt_img

    def get_img_from_index(self, index):
        index, dataset_index = self.get_correct_index(index)
        img_id = self.img_ids[dataset_index][index]
        coco = self.cocos[dataset_index]

        img = self.load_img(coco, img_id)

        img = self.transforms(image=img)["image"]
        factor = random.uniform(0.3,0.5)
        img = img.float()
        img = img * (factor * 255.0 / img.mean())
        img = img.round().clamp(0, 255).to(torch.uint8)

        return img

    def load_img(self, coco: COCO, img_id: int) -> np.ndarray:
        img_info = coco.loadImgs(img_id)[0]
        file_path = img_info["file_name"]
        img = cv2.imread(
            os.path.join(self.exlpose_root, file_path), cv2.IMREAD_COLOR_RGB
        )
        return img

    def shuffle_img_ids(self):
        self.normal_selected_img_ids = random.sample(self.normal_img_ids, 120)
        self.hard_selected_img_ids = random.sample(self.hard_img_ids, 120)
        self.extreme_selected_img_ids = random.sample(self.extreme_img_ids, 120)
        self.a7m3_selected_img_ids = random.sample(self.a7m3_img_ids, 180)
        self.ricoh3_selected_img_ids = random.sample(self.ricoh3_img_ids, 180)

        self.img_ids = [
            self.normal_selected_img_ids,
            self.hard_selected_img_ids,
            self.extreme_selected_img_ids,
            self.a7m3_selected_img_ids,
            self.ricoh3_selected_img_ids,
        ]


def get_dataloader(cfg):
    return DataLoader(
        TrainDataset(cfg),
        cfg.data.batch_size,
        cfg.data.shuffle,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        persistent_workers=cfg.data.persistent_workers,
    )
