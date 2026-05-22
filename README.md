# UDAPose
Official repo of CVPR'26 paper "UDAPose: Unsupervised Domain Adaptation for Low-Light Human Pose Estimation"

## Roadmap

- [x] Inference code and model weight
- [ ] Pose model training and synthetic data
- [ ] Data synthesis pipeline and training

## Environment

```bash
uv sync
source .venv/bin/activate
```

```bash
cd models/edpose/ops
python setup.py build install
```

```bash
python test.py
```

## Inference

1. Download checkpoints.

```bash
hf download arsity/UDAPose-model-weights --local-dir ckpts
```

2. Download [ExLPose](https://github.com/sohyun-l/ExLPose) dataset and put into `data`.

3. Organize the dataset as following:

```bash
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
