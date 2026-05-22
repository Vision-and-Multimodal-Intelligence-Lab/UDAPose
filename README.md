# UDAPose
Official repo of CVPR'26 paper "UDAPose: Unsupervised Domain Adaptation for Low-Light Human Pose Estimation".

## Roadmap

- [x] Inference code and model weights
- [ ] Pose model training and synthetic data
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
         |   |   |- (A7M3 images)...
    ```

4. Run inference

    ```bash
    sh test.sh
    ```
