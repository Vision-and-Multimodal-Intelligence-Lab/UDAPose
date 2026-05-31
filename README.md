# UDAPose
> UDAPose: Unsupervised Domain Adaptation for Low-Light Human Pose Estimation, *CVPR 2026*  
> [![arXiv](https://img.shields.io/badge/arXiv-2604.10485-b31b1b.svg?style=flat)](https://arxiv.org/abs/2604.10485)

## Roadmap

- [x] Inference code and model weights
- [x] Pose model training and synthetic data
- [ ] Data synthesis pipeline and training

## Environment

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

## Inference

1. Download checkpoints from [🤗](https://huggingface.co/arsity/UDAPose-model-weights).

    ```bash
    hf download arsity/UDAPose-model-weights --local-dir ckpts
    ```

2. Download [ExLPose](https://github.com/sohyun-l/ExLPose) dataset and put into `data`.

3. Organize the dataset as following:

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

4. Run inference

    ```bash
    sh test.sh
    ```

## Train

### Train Pose Model

Download synthetic data from [🤗](https://huggingface.co/datasets/arsity/UDAPose-synthetic-data).

```bash
hf download arsity/UDAPose-synthetic-data images.zip mapping_list.json --type dataset --local-dir data
```

Unzip and organize as following

```
data
 |- mapping_list.json
 |- synthetic
     |- 0
     |- 1
     |- (image id directories)...
```

then

```bash
sh train.sh
```

to start training (for low-light). If you want to start from scratch (well-lit), you can edit `train.sh`.

## Bibtex

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
