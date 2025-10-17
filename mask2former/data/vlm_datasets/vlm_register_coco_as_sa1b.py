# Copyright (c) Facebook, Inc. and its affiliates.
import json
import os

from .vlm_classes import get_sa1b_categories_with_prompt_eng

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.file_io import PathManager
import logging


SA1B_CATEGORIES = get_sa1b_categories_with_prompt_eng()
SA1B_COLORS = [k["color"] for k in SA1B_CATEGORIES]

MetadataCatalog.get("sa1b_segment_train").set(
    stuff_colors=SA1B_COLORS[:],
)
MetadataCatalog.get("sa1b_segment_train").set(
    thing_colors=SA1B_COLORS[:],
)

_logger = logging.getLogger(__name__)


def load_sa1b_segments(image_dir, annotations_dir, meta):
    """
    Args:
        image_dir (str): path to the raw dataset. e.g., "~/coco/train2017".
        gt_dir (str): path to the raw annotations. e.g., "~/coco/panoptic_train2017".
        json_file (str): path to the json file. e.g., "~/coco/annotations/panoptic_train2017.json".
    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )
    """
    ret = []

    annotations = sorted(os.listdir(annotations_dir))
    images = sorted(os.listdir(image_dir))

    for ann, img in zip(annotations, images):
        image_id_ann = ann.split(".")[0]
        image_id_img = img.split(".")[0]
        assert (
            image_id_ann == image_id_img
        ), f"Image and annotation mismatch: {image_id_ann} != {image_id_img}"
        image_file = os.path.join(image_dir, img)
        ann_file = os.path.join(annotations_dir, ann)
        ret.append(
            {
                "file_name": image_file,
                "image_id": image_id_ann,
                "ann_file": ann_file,
            }
        )

    assert len(ret), f"No images found in {image_dir}!"
    _logger.info(f"Loaded {len(ret)} images from {image_dir}")

    assert PathManager.isfile(ret[0]["file_name"]), ret[0]["file_name"]
    assert PathManager.isfile(ret[0]["ann_file"]), ret[0]["ann_file"]
    return ret


def register_sa1b_segments(
    name,
    metadata,
    image_root,
    annotations_root,
):
    panoptic_name = name
    DatasetCatalog.register(
        panoptic_name,
        lambda: load_sa1b_segments(image_root, annotations_root, metadata),
    )
    MetadataCatalog.get(panoptic_name).set(
        image_root=image_root,
        annotations_root=annotations_root,
        evaluator_type="ade20k_panoptic_seg",
        ignore_label=255,
        label_divisor=1000,
        **metadata,
    )


_PREDEFINED_SPLITS_SA1B_SEGMENTS = {
    "vlm_coco_whole_sa1b_train": (
        "coco/train2017",
        "sam_labels_coco/instance_anns",
    ),
}


def get_metadata():
    meta = {}
    # The following metadata maps contiguous id from [0, #thing categories +
    # #stuff categories) to their names and colors. We have to replica of the
    # same name and color under "thing_*" and "stuff_*" because the current
    # visualization function in D2 handles thing and class classes differently
    # due to some heuristic used in Panoptic FPN. We keep the same naming to
    # enable reusing existing visualization functions.
    thing_classes = [k["name"] for k in SA1B_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in SA1B_CATEGORIES if k["isthing"] == 1]
    stuff_classes = [k["name"] for k in SA1B_CATEGORIES]
    stuff_colors = [k["color"] for k in SA1B_CATEGORIES]

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

    for i, cat in enumerate(SA1B_CATEGORIES):
        if cat["isthing"]:
            thing_dataset_id_to_contiguous_id[cat["id"]] = i
        # else:
        #     stuff_dataset_id_to_contiguous_id[cat["id"]] = i

        # in order to use sem_seg evaluator
        stuff_dataset_id_to_contiguous_id[cat["id"]] = i

    meta["thing_dataset_id_to_contiguous_id"] = thing_dataset_id_to_contiguous_id
    meta["stuff_dataset_id_to_contiguous_id"] = stuff_dataset_id_to_contiguous_id

    return meta


def register_vlm_sa1b_segments(root):
    metadata = get_metadata()
    for (
        prefix,
        (image_root, annotations_root),
    ) in _PREDEFINED_SPLITS_SA1B_SEGMENTS.items():
        # The "standard" version of COCO panoptic segmentation dataset,
        # e.g. used by Panoptic-DeepLab
        register_sa1b_segments(
            prefix,
            metadata,
            os.path.join(root, image_root),
            os.path.join(root, annotations_root),
        )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_vlm_sa1b_segments(_root)
