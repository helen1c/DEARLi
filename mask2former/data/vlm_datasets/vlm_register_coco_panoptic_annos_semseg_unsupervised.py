"""
This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”).
All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates. 

Reference: https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/data/datasets/register_coco_panoptic_annos_semseg.py
"""

import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg
from .vlm_classes import get_coco_categories_with_prompt_eng
from detectron2.utils.file_io import PathManager
import logging

logger = logging.getLogger(__name__)

COCO_CATEGORIES = get_coco_categories_with_prompt_eng()

_PREDEFINED_SPLITS_COCO_PANOPTIC = {
    f"vlm_coco_2017_train_panoptic_semi_sup_unlabeled_train_{split}": (
        "coco/train2017",
        f"coco_splits/{split}/unlabeled.txt",
    )
    for split in ["1_512", "1_256", "1_128", "1_64", "1_32"]
}


def get_metadata():
    meta = {}
    thing_classes = [k["name"] for k in COCO_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in COCO_CATEGORIES if k["isthing"] == 1]
    stuff_classes = [k["name"] for k in COCO_CATEGORIES]
    stuff_colors = [k["color"] for k in COCO_CATEGORIES]

    meta["thing_classes"] = thing_classes
    meta["thing_colors"] = thing_colors
    meta["stuff_classes"] = stuff_classes
    meta["stuff_colors"] = stuff_colors

    thing_dataset_id_to_contiguous_id = {}
    stuff_dataset_id_to_contiguous_id = {}
    contiguous_id_to_class_name = []

    for i, cat in enumerate(COCO_CATEGORIES):
        if cat["isthing"]:
            thing_dataset_id_to_contiguous_id[cat["id"]] = i
        # else:
        #     stuff_dataset_id_to_contiguous_id[cat["id"]] = i

        # in order to use sem_seg evaluator
        stuff_dataset_id_to_contiguous_id[cat["id"]] = i

        contiguous_id_to_class_name.append(cat["name"])

    meta["thing_dataset_id_to_contiguous_id"] = thing_dataset_id_to_contiguous_id
    meta["stuff_dataset_id_to_contiguous_id"] = stuff_dataset_id_to_contiguous_id
    meta["contiguous_id_to_class_name"] = contiguous_id_to_class_name

    return meta


def load_coco_panoptic_json(image_dir, meta, split_file_path):
    """
    Args:
        image_dir (str): path to the raw dataset. e.g., "~/coco/train2017".
        gt_dir (str): path to the raw annotations. e.g., "~/coco/panoptic_train2017".
        json_file (str): path to the json file. e.g., "~/coco/annotations/panoptic_train2017.json".
    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )
    """
    if split_file_path is not None:
        with open(split_file_path, "r") as f:
            split_info = f.readlines()
            split_info = [x.strip() for x in split_info]
            split_info = [
                x.split()[0].strip().split("/")[-1][: -len(".jpg")] for x in split_info
            ]

    ret = []
    for image_id in split_info:
        image_file = os.path.join(image_dir, image_id + ".jpg")
        ret.append(
            {
                "file_name": image_file,
                "image_id": image_id,
            }
        )

    assert len(ret), f"No images found in {image_dir}!"
    logger.info(f"Loaded {len(ret)} images from {image_dir}")
    assert PathManager.isfile(ret[0]["file_name"]), ret[0]["file_name"]
    return ret


def register_coco_panoptic_annos_sem_seg(
    name,
    metadata,
    image_root,
    split_file_path,
):
    panoptic_name = name
    # delattr(MetadataCatalog.get(panoptic_name), "thing_classes")
    # delattr(MetadataCatalog.get(panoptic_name), "thing_colors")
    MetadataCatalog.get(panoptic_name).set(
        thing_classes=metadata["thing_classes"],
        thing_colors=metadata["thing_colors"],
        # thing_dataset_id_to_contiguous_id=metadata["thing_dataset_id_to_contiguous_id"],
    )

    # the name is "coco_2017_train_panoptic_with_sem_seg" and "coco_2017_val_panoptic_with_sem_seg"
    semantic_name = name + "_with_sem_seg"
    DatasetCatalog.register(
        semantic_name,
        lambda: load_coco_panoptic_json(
            image_root,
            metadata,
            split_file_path,
        ),
    )
    MetadataCatalog.get(semantic_name).set(
        image_root=image_root,
        evaluator_type="coco_panoptic_seg",
        ignore_label=255,
        label_divisor=1000,
        **metadata,
    )


def register_unsupervised_coco_panoptic(root):
    for (
        prefix,
        (image_root, split_file),
    ) in _PREDEFINED_SPLITS_COCO_PANOPTIC.items():

        register_coco_panoptic_annos_sem_seg(
            prefix,
            get_metadata(),
            os.path.join(root, image_root),
            os.path.join(root, split_file),
        )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_unsupervised_coco_panoptic(_root)
