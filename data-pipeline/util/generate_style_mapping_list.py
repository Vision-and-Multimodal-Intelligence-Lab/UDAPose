"""
Generate style mapping list for each partition to ensure reproducibility.
"""

import json
import random
from os import path

from crowdposetools.coco import COCO
from tqdm import tqdm

EXLPOSE_ROOT = "/datapool/datasets/ExLPose"
WL_TRAIN_COCO = COCO(path.join(EXLPOSE_ROOT, "Annotations", "ExLPose_train_WL.json"))
WL_TEST_COCO = COCO(path.join(EXLPOSE_ROOT, "Annotations", "ExLPose_test_WL.json"))

LL_N_COCO = COCO(path.join(EXLPOSE_ROOT, "Annotations", "ExLPose_test_LL-N.json"))
LL_H_COCO = COCO(path.join(EXLPOSE_ROOT, "Annotations", "ExLPose_test_LL-H.json"))
LL_E_COCO = COCO(path.join(EXLPOSE_ROOT, "Annotations", "ExLPose_test_LL-E.json"))
A7M3_COCO = COCO(path.join(EXLPOSE_ROOT, "Annotations", "ExLPose-OC_test_A7M3.json"))
RICOH3_COCO = COCO(
    path.join(EXLPOSE_ROOT, "Annotations", "ExLPose-OC_test_RICOH3.json")
)

# Number of candidates for each partition
CANDIDATES = 4


def main():
    wl_ids = WL_TRAIN_COCO.getImgIds() + WL_TEST_COCO.getImgIds()
    ll_n_ids = sorted(LL_N_COCO.getImgIds())
    ll_h_ids = sorted(LL_H_COCO.getImgIds())
    ll_e_ids = sorted(LL_E_COCO.getImgIds())
    a7m3_ids = sorted(A7M3_COCO.getImgIds())
    ricoh3_ids = sorted(RICOH3_COCO.getImgIds())

    res = {}

    for wl_img_id in tqdm(wl_ids):
        res[wl_img_id] = {}

        res[wl_img_id]["normal"] = random.sample(ll_n_ids, CANDIDATES)
        res[wl_img_id]["hard"] = random.sample(ll_h_ids, CANDIDATES)
        res[wl_img_id]["extreme"] = random.sample(ll_e_ids, CANDIDATES)
        res[wl_img_id]["a7m3"] = random.sample(a7m3_ids, CANDIDATES)
        res[wl_img_id]["ricoh3"] = random.sample(ricoh3_ids, CANDIDATES)

    with open("style_mapping_list.json", "w") as f:
        json.dump(res, f, indent=4)


if __name__ == "__main__":
    main()
