"""
Dataset (Dataloader) for style transfer.
In this file, we would use a combination of LL-N, LL-H, LL-E, A7M3, RICOH3 to generate low-light images.
We combine all normalization methods into one dataloader.
"""

import json
import os
from os import path

import cv2
import einops
import torch
from crowdposetools.coco import COCO
from torch.utils.data import DataLoader, Dataset

from util.suppress_stdout import suppress_stdout


class ExLPoseDataset(Dataset):
    """
    Dataset (Dataloader) for style transfer.
    Use a combination of LL-N, LL-H, LL-E, A7M3, RICOH3 as style references.
    Use well-lit images as content references.
    """

    def __init__(self, cfg, **kwargs):
        super().__init__(**kwargs)
        assert os.path.exists(cfg.dataset.exlpose_root), (
            f"{cfg.dataset.exlpose_root} does not exist."
        )
        assert os.path.exists(cfg.dataset.mapping_config), (
            f"{cfg.dataset.mapping_config} does not exist."
        )

        self.cfg = cfg
        self.exlpose_root = cfg.dataset.exlpose_root
        self.image_resize = cfg.dataset.image_resize
        with open(cfg.dataset.mapping_config, "r") as f:
            self.mapping_config = json.load(f)

        self.cur_epoch = 0

        with suppress_stdout():
            self.wl_coco = COCO(
                os.path.join(self.exlpose_root, "Annotations", "ExLPose_train_WL.json")
                # os.path.join(self.exlpose_root, "Annotations", "ExLPose_test_WL.json")
            )
            self.normal_coco = COCO(
                os.path.join(self.exlpose_root, "Annotations", "ExLPose_test_LL-N.json")
                # os.path.join(self.exlpose_root, "Annotations", "ExLPose_test_LL-A.json")
                # os.path.join(self.exlpose_root, "Annotations", "ExLPose_train_LL.json")
                # os.path.join(self.exlpose_root, "Annotations", "ExLPose_test_WL.json")

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

        self.wl_img_ids = sorted(self.wl_coco.getImgIds())
        self.normal_img_ids = sorted(self.normal_coco.getImgIds())
        self.hard_img_ids = sorted(self.hard_coco.getImgIds())
        self.extreme_img_ids = sorted(self.extreme_coco.getImgIds())
        self.a7m3_img_ids = sorted(self.a7m3_coco.getImgIds())
        self.ricoh3_img_ids = sorted(self.ricoh3_coco.getImgIds())

    def __len__(self):
        return len(self.wl_img_ids)
        # return len(self.normal_img_ids)

    def __getitem__(self, index):
        # load well-lit as content reference
        wl_img_id = self.wl_img_ids[index]
        # wl_img_id = self.normal_img_ids[index]
        wl_img, wl_fname, (wl_h, wl_w) = self.load_image(wl_img_id, self.wl_coco)

        # Note: follow the order of mapping_config
        normal_img_id = self.mapping_config[str(wl_img_id)]["normal"][self.cur_epoch]
        # normal_img_id = wl_img_id
        hard_img_id = self.mapping_config[str(wl_img_id)]["hard"][self.cur_epoch]
        extreme_img_id = self.mapping_config[str(wl_img_id)]["extreme"][self.cur_epoch]
        a7m3_img_id = self.mapping_config[str(wl_img_id)]["a7m3"][self.cur_epoch]
        ricoh3_img_id = self.mapping_config[str(wl_img_id)]["ricoh3"][self.cur_epoch]

        # load low-light as style reference
        normal_img, normal_fname, (normal_h, normal_w) = self.load_image(
            normal_img_id, self.normal_coco
        )
        hard_img, hard_fname, (hard_h, hard_w) = self.load_image(
            hard_img_id, self.hard_coco
        )
        extreme_img, extreme_fname, (extreme_h, extreme_w) = self.load_image(
            extreme_img_id, self.extreme_coco
        )
        a7m3_img, a7m3_fname, (a7m3_h, a7m3_w) = self.load_image(
            a7m3_img_id, self.a7m3_coco
        )
        ricoh3_img, ricoh3_fname, (ricoh3_h, ricoh3_w) = self.load_image(
            ricoh3_img_id, self.ricoh3_coco
        )
        # hard_img, hard_fname, (hard_h, hard_w) = normal_img, normal_fname, (normal_h, normal_w)
        # extreme_img, extreme_fname, (extreme_h, extreme_w) = normal_img, normal_fname, (normal_h, normal_w)
        # a7m3_img, a7m3_fname, (a7m3_h, a7m3_w) = normal_img, normal_fname, (normal_h, normal_w)
        # ricoh3_img, ricoh3_fname, (ricoh3_h, ricoh3_w) = normal_img, normal_fname, (normal_h, normal_w)

        return (
            {
                "img": wl_img,
                "fname": wl_img_id,
                "h": wl_h,
                "w": wl_w,
            },
            {
                "img": normal_img,
                "fname": normal_img_id,
                "h": normal_h,
                "w": normal_w,
            },
            {
                "img": hard_img,
                "fname": hard_img_id,
                "h": hard_h,
                "w": hard_w,
            },
            {
                "img": extreme_img,
                "fname": extreme_img_id,
                "h": extreme_h,
                "w": extreme_w,
            },
            {
                "img": a7m3_img,
                "fname": a7m3_img_id,
                "h": a7m3_h,
                "w": a7m3_w,
            },
            {
                "img": ricoh3_img,
                "fname": ricoh3_img_id,
                "h": ricoh3_h,
                "w": ricoh3_w,
            },
        )

    def load_image(self, img_id, coco):
        img_info = coco.loadImgs(img_id)[0]
        img_path = path.join(self.exlpose_root, img_info["file_name"])
        h, w = img_info["height"], img_info["width"]
        fname = path.splitext(path.basename(img_path))[0]

        img = cv2.imread(img_path, cv2.IMREAD_COLOR_RGB)
        img = cv2.resize(img, (self.image_resize[1], self.image_resize[0]))
        img = torch.from_numpy(img)
        img = einops.rearrange(img, "h w c -> c h w")

        return img, fname, (h, w)


def get_dataloader(cfg):
    return DataLoader(
        ExLPoseDataset(cfg),
        cfg.dataset.batch_size,
        cfg.dataset.shuffle,
        num_workers=cfg.dataset.num_workers,
        pin_memory=cfg.dataset.pin_memory,
    )
