# UDAPose

Official implementation of the paper

> UDAPose: Unsupervised Domain Adaptation for Low-Light Human Pose Estimation, *CVPR 2026*  
> [![CVF Open Access](https://img.shields.io/badge/CVF%20Open%20Access-CVPR%202026-7898CA.svg?style=flat)](https://openaccess.thecvf.com/content/CVPR2026/html/Chen_UDAPose_Unsupervised_Domain_Adaptation_for_Low-Light_Human_Pose_Estimation_CVPR_2026_paper.html) [![arXiv](https://img.shields.io/badge/arXiv-2604.10485-b31b1b.svg?style=flat)](https://arxiv.org/abs/2604.10485)

![teaser](./img/teaser.png)

## 🛠️ Setup

### Python environment

1. If you use [uv](https://docs.astral.sh/uv/), you can directly

    ```bash
    uv sync
    ```

    Or you can still use `pip` as following

    ```bash
    pip install -r requirements.txt
    ```

2. After that, you should compile and install the CUDA kernel for `deformable attention`.

    ```bash
    cd models/edpose/ops
    python setup.py build install
    ```

3. Test installation.
    ```bash
    python test.py
    ```

### Prepare ExLPose dataset

1. Download [ExLPose](https://github.com/sohyun-l/ExLPose) dataset and put it under `data`.

2. Organize the dataset as following:

    ```
    data
     |- ExLPose
         |- Annotations
         |   |- ExLPose_test_WL.json
         |   |- ExLPose-OC_test_A7M3.json
         |   |- ...
         |- bright
         |   |- (bright images)...
         |- dark
         |   |- (paired dark images)...
         |- ExLPose-OCN
         |   |- A7M3
         |   |   |- (A7M3 images)...
         |   |- RICOH3
         |   |   |- (RICOH3 images)...
    ```

## 🕵🏼 Inference

1. Download checkpoints from [🤗](https://huggingface.co/arsity/UDAPose-model-weights).

    ```bash
    hf download arsity/UDAPose-model-weights --local-dir ckpts
    ```

2. Run inference

    ```bash
    sh test.sh
    ```

You may also want to edit `test.sh` to evaluate on one subset.

## 🏋🏼 Train

Our full framework involves 3 steps in total. You can start from any step with our provided checkpoints or from the very beginning.

### Train LCIM

1. Download SD 2.1 checkpoints from [🤗](https://huggingface.co/arsity/UDAPose-model-weights).

    ```bash
    hf download arsity/UDAPose-model-weights --local-dir ckpts
    ```

2. Start training LCIM

    ```bash
    cd data-pipeline
    python train_vae.py
    ```

Results would be under `ckpts/vae_train_outputs`.

### Generate training data

1. Download checkpoints from [🤗](https://huggingface.co/arsity/UDAPose-model-weights).

    ```bash
    hf download arsity/UDAPose-model-weights --local-dir ckpts
    ```

2. Start generating training data

    ```bash
    cd data-pipeline
    python style_transfer.py
    ```

Synthetic training data would be under `data/synthetic`.

### Train Pose Model

1. Download checkpoints from [🤗](https://huggingface.co/arsity/UDAPose-model-weights).

    ```bash
    hf download arsity/UDAPose-model-weights --local-dir ckpts
    ```

2. Download synthetic data from [🤗](https://huggingface.co/datasets/arsity/UDAPose-synthetic-data).

    ```bash
    hf download arsity/UDAPose-synthetic-data images.zip mapping_list.json --type dataset --local-dir data
    ```

3. Unzip and organize as following

    ```
    data
    |- mapping_list.json
    |- synthetic
        |- 0
        |- 1
        |- (image id directories)...
    ```

4. then

    ```bash
    sh train.sh
    ```

    to start training (for low-light). If you want to start from scratch (well-lit), you can edit `train.sh`.

## ©️ License

UDAPose is released under the **Apache License 2.0** for our original contributions, unless otherwise noted.

This project builds upon several open-source projects. We preserve their original license notices, including:

- [ED-Pose](https://github.com/IDEA-Research/ED-Pose): Apache License 2.0, with the original ED-Pose license notices retained
- [StyleID](https://github.com/jiwoogit/StyleID): MIT License

Some optional components, pretrained models, or external checkpoints may be subject to their own licenses, such as [Stable Diffusion](https://huggingface.co/sd2-community/stable-diffusion-2-1) / [SwinTransformer](https://github.com/microsoft/Swin-Transformer). These are not covered by the Apache-2.0 license of our original code.

Please see `LICENSES/` for details.

## 📝 Bibtex

If you find this work useful, please consider cite our paper

```bibtex
@InProceedings{chen2026udapose,
    author    = {Chen, Haopeng and Ai, Yihao and Kim, Kabeen and Tan, Robby T. and Chen, Yixin and Wang, Bo},
    title     = {UDAPose: Unsupervised Domain Adaptation for Low-Light Human Pose Estimation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2026},
    pages     = {13781-13792}
}
```
