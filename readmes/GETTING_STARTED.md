# Getting Started with DEARLi

This document provides a concise introduction to using **DEARLi** — including installation, dataset preparation, checkpoints, training, and evaluation.

For additional background, refer to:
- [Getting Started with Detectron2](https://github.com/facebookresearch/detectron2/blob/master/GETTING_STARTED.md)
- [INSTALL.md](./INSTALL.md)
- [DATASETS.md](./DATASETS.md)

---

## Installation

Follow the detailed steps in **[INSTALL.md](./INSTALL.md)** to set up your environment and dependencies.

## Datasets Preparation

Please refer to **[DATASETS.md](./DATASETS.md)** for instructions on downloading and organizing the required datasets (**COCO** and **ADE20K**).

Ensure your dataset directory structure follows the conventions described there.

---

## Checkpoints

All **30 pretrained checkpoints**, along with **2 decoder warmup weights**, are available at the following OneDrive link:  
🔗 [Download Checkpoints](https://ferhr-my.sharepoint.com/:f:/g/personal/imartinovic_fer_hr/Ekllam9FWTdGnCUAdF4Pc1oB0ifhoTe407oELLxHgsUyLw?e=f9jA2X)

After downloading, create a `checkpoints` directory in the project root and **maintain the same folder structure** as provided on OneDrive:

```
checkpoints/
  decoder_warmup_weights/
    *.pth
  coco-panoptic_weights/
    *.pth
  coco-objects_weights/
    *.pth
  ade20k_weights/
    *.pth
```


## Training

To train models on any of the supported datasets, use the script:

```bash
./scripts/train.sh <dataset> <split> <method>
```

### Arguments

| Argument | Options | Description |
|-----------|----------|-------------|
| **dataset** | `ade20k`, `coco-obj`, `coco-pan` | Target dataset for training |
| **split** | ADE20K → `1_128`, `1_64`, `1_32`, `1_16`, `1_8`<br>COCO → `1_512`, `1_256`, `1_128`, `1_64`, `1_32` | Defines the labeled data ratio used for semi-supervised training |
| **method** | `dear`, `dearli` | Training method; `dearli` adds  additional weights |

### Example

```bash
./scripts/train_dearli.sh ade20k 1_128 dear
```


##  Evaluation

To evaluate pretrained or custom-trained checkpoints, use:

```bash
./scripts/eval.sh <dataset> <split> <checkpoint_path> [num_gpus]
```

### Arguments

| Argument | Description |
|-----------|-------------|
| **dataset** | `ade20k`, `coco-obj`, or `coco-pan` |
| **split** | ADE20K → `1_128`, `1_64`, `1_32`, `1_16`, `1_8`<br>COCO → `1_512`, `1_256`, `1_128`, `1_64`, `1_32` |
| **checkpoint_path** | Path to the model checkpoint (`.pth`) to evaluate |
| **num_gpus** *(optional)* | Number of GPUs to use (default: `1`) |

### Example

```bash
./scripts/eval_dearli.sh coco-pan 1_128 checkpoints/coco-panoptic_weights/DEAR_coco_pan_1_128.pth 4
```

This script automatically locates the correct config file and runs evaluation in `--eval-only` mode.


## Summary

| Step | Description |
|------|--------------|
| **1.** | Install environment dependencies using [INSTALL.md](./INSTALL.md) |
| **2.** | Prepare COCO and ADE20K datasets using [DATASETS.md](./DATASETS.md) |
| **3.** | Download checkpoints from the [OneDrive link](https://ferhr-my.sharepoint.com/:f:/g/personal/imartinovic_fer_hr/Ekllam9FWTdGnCUAdF4Pc1oB0ifhoTe407oELLxHgsUyLw?e=f9jA2X) |
| **4.** | Train models using `scripts/train_dearli.sh` |
| **5.** | Evaluate models using `scripts/eval_dearli.sh` |
