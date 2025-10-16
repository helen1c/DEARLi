import json
from pathlib import Path

import numpy as np
from detectron2.data import detection_utils as utils
from panopticapi.utils import id2rgb, rgb2id
from PIL import Image
from tqdm import tqdm

THING_MAX_ID = 90


def ensure_background_category(annotations):
    cats = annotations.get("categories", [])
    has_bg = any(c.get("id") == 0 for c in cats)

    # if there is a background category, something is wrong
    if has_bg:
        print("Warning: annotations already have a category with id 0")
        raise ValueError("Category with id 0 already exists")

    # COCO-style background placeholder
    print("Adding background category with id 0")
    cats.append(
        {"id": 0, "name": "background", "supercategory": "background", "isthing": 0}
    )
    annotations["categories"] = cats


for spl in ["train", "val"]:
    print("Processing", spl)
    output_folder = Path(f"./datasets/coco/panoptic_{spl}2017_objects")
    output_folder.mkdir(parents=True, exist_ok=True)

    json_path = f"./datasets/coco/annotations/panoptic_{spl}2017.json"
    with open(json_path, "r") as f:
        annotations = json.load(f)

    ensure_background_category(annotations)

    panoptic_root = Path(f"./datasets/coco/panoptic_{spl}2017")

    for annotation in tqdm(
        annotations["annotations"], total=len(annotations["annotations"])
    ):
        image_file = annotation["file_name"]
        panoptic_path = panoptic_root / image_file

        if not panoptic_path.exists():
            # Skip missing files instead of crashing
            continue

        panoptic_img = utils.read_image(str(panoptic_path), format="RGB")
        panoptic_id = rgb2id(panoptic_img).astype(np.int32)

        segments_infos_old = annotation["segments_info"]
        segments_info = []
        background_segment = None
        unique_ids = set()

        # Split: keep thing (<=90), merge stuff (>90) into one bg segment (id of first stuff)
        for seg_info in segments_infos_old:
            unique_ids.add(seg_info["id"])
            if seg_info["category_id"] <= THING_MAX_ID:
                segments_info.append(seg_info)
            else:
                if background_segment is None:
                    # clone seg_info to start bg
                    background_segment = dict(seg_info)
                    background_segment["category_id"] = 0
                else:
                    # merge areas and remap ids to bg id
                    background_segment["area"] += seg_info["area"]
                    panoptic_id[panoptic_id == seg_info["id"]] = background_segment[
                        "id"
                    ]

        zero_mask = panoptic_id == 0

        if background_segment is not None:
            # add void/zero pixels to bg
            if zero_mask.any():
                background_segment["area"] += int(zero_mask.sum())
                panoptic_id[zero_mask] = background_segment["id"]
            segments_info.append(background_segment)
        else:
            # no stuff segments existed; if zeros exist, create a fresh bg id not in use
            if zero_mask.any():
                n_id = 1
                while n_id in unique_ids:
                    n_id += 1
                panoptic_id[zero_mask] = n_id
                segments_info.append(
                    {
                        "id": n_id,
                        "category_id": 0,
                        "area": int(zero_mask.sum()),
                        "iscrowd": 0,
                    }
                )

        # Save remapped RGB panoptic image
        panoptic_img_new = id2rgb(panoptic_id)
        Image.fromarray(panoptic_img_new).save(output_folder / image_file)

        # Update annotation segments
        annotation["segments_info"] = segments_info

    new_json_path = f"./datasets/coco/annotations/panoptic_{spl}2017_objects.json"
    print("Saving to", new_json_path)
    with open(new_json_path, "w") as f:
        json.dump(annotations, f, indent=2)
