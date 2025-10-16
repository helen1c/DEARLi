# Copyright (c) Facebook, Inc. and its affiliates.
import json
import os

from .vlm_classes import get_ade20k_categories_with_prompt_eng

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.file_io import PathManager

ADE20K_150_CATEGORIES = get_ade20k_categories_with_prompt_eng()
ADE20K_COLORS = [k["color"] for k in ADE20K_150_CATEGORIES]

MetadataCatalog.get("ade20k_sem_seg_train").set(
    stuff_colors=ADE20K_COLORS[:],
)

MetadataCatalog.get("ade20k_sem_seg_val").set(
    stuff_colors=ADE20K_COLORS[:],
)
import logging

logger = logging.getLogger(__name__)


def load_semisup_ade20k_panoptic_json(image_dir, meta, split_file):
    """
    Args:
        image_dir (str): path to the raw dataset. e.g., "~/coco/train2017".
        gt_dir (str): path to the raw annotations. e.g., "~/coco/panoptic_train2017".
        json_file (str): path to the json file. e.g., "~/coco/annotations/panoptic_train2017.json".
    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )
    """

    if split_file is not None:
        with open(split_file, "r") as f:
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


def register_semisup_ade20k_panoptic(
    name,
    metadata,
    image_root,
    split_file=None,
):
    """
    Register a "standard" version of ADE20k panoptic segmentation dataset named `name`.
    The dictionaries in this registered dataset follows detectron2's standard format.
    Hence it's called "standard".
    Args:
        name (str): the name that identifies a dataset,
            e.g. "ade20k_panoptic_train"
        metadata (dict): extra metadata associated with this dataset.
        image_root (str): directory which contains all the images
        panoptic_root (str): directory which contains panoptic annotation images in COCO format
        panoptic_json (str): path to the json panoptic annotation file in COCO format
        sem_seg_root (none): not used, to be consistent with
            `register_coco_panoptic_separated`.
        instances_json (str): path to the json instance annotation file
    """
    assert split_file is not None or not os.path.exists(
        split_file
    ), f"split_file: [{split_file}] is None or does not exist"
    panoptic_name = name
    DatasetCatalog.register(
        panoptic_name,
        lambda: load_semisup_ade20k_panoptic_json(
            image_root,
            metadata,
            split_file,
        ),
    )
    MetadataCatalog.get(panoptic_name).set(
        image_root=image_root,
        evaluator_type="ade20k_panoptic_seg",
        ignore_label=255,
        label_divisor=1000,
        **metadata,
    )


_PREDEFINED_SPLITS_ADE20K_PANOPTIC = {
    f"vlm_ade20k_panoptic_semi_sup_unlabeled_train_{split}": (
        "ADEChallengeData2016/images/training",
        f"ade_splits/{split}/unlabeled.txt",
    )
    for split in ["1_8", "1_16", "1_32", "1_64", "1_128"]
}


def get_metadata():
    meta = {}
    # The following metadata maps contiguous id from [0, #thing categories +
    # #stuff categories) to their names and colors. We have to replica of the
    # same name and color under "thing_*" and "stuff_*" because the current
    # visualization function in D2 handles thing and class classes differently
    # due to some heuristic used in Panoptic FPN. We keep the same naming to
    # enable reusing existing visualization functions.
    thing_classes = [k["name"] for k in ADE20K_150_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in ADE20K_150_CATEGORIES if k["isthing"] == 1]
    stuff_classes = [k["name"] for k in ADE20K_150_CATEGORIES]
    stuff_colors = [k["color"] for k in ADE20K_150_CATEGORIES]

    meta["thing_classes"] = thing_classes
    meta["thing_colors"] = thing_colors
    meta["stuff_classes"] = stuff_classes
    meta["stuff_colors"] = stuff_colors

    # Convert category id for training:
    #   category id: like semantic segmentation, it is the class id for each
    #   pixel. Since there are some classes not used in evaluation, the category
    #   id is not always contiguous and thus we have two set of category ids:
    #       - original category id: category id in the original dataset, mainly
    #           used for evaluation.
    #       - contiguous category id: [0, #classes), in order to train the linear
    #           softmax classifier.
    thing_dataset_id_to_contiguous_id = {}
    stuff_dataset_id_to_contiguous_id = {}

    for i, cat in enumerate(ADE20K_150_CATEGORIES):
        if cat["isthing"]:
            thing_dataset_id_to_contiguous_id[cat["id"]] = i
        # else:
        #     stuff_dataset_id_to_contiguous_id[cat["id"]] = i

        # in order to use sem_seg evaluator
        stuff_dataset_id_to_contiguous_id[cat["id"]] = i

    meta["thing_dataset_id_to_contiguous_id"] = thing_dataset_id_to_contiguous_id
    meta["stuff_dataset_id_to_contiguous_id"] = stuff_dataset_id_to_contiguous_id

    return meta


def register_vlm_all_semisup_unsupervised_ade20k_panoptic(root):
    metadata = get_metadata()
    for (
        prefix,
        (
            image_root,
            split_file,
        ),
    ) in _PREDEFINED_SPLITS_ADE20K_PANOPTIC.items():
        register_semisup_ade20k_panoptic(
            prefix,
            metadata,
            os.path.join(root, image_root),
            os.path.join(root, split_file),
        )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_vlm_all_semisup_unsupervised_ade20k_panoptic(_root)
