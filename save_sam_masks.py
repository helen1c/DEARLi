# Iterate through the whole folder, load image, predict masks with sam2 and save them on disk
# Usage: python save_sam2_masks.py --input_folder path/to/folder --output_folder path/to/folder
# Example: python save_sam2_masks.py --input_folder /home/alex/Downloads/ISIC2018_Task1-2_Training_Input --output_folder /home/alex/Downloads/ISIC2018_Task1-2_Training_Input_masks
import argparse
from itertools import product
from pathlib import Path

import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from panopticapi.utils import id2rgb
import tqdm
from pycocotools import mask as mask_utils

import torch
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def show_anns(anns, borders=True):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones(
        (
            sorted_anns[0]["segmentation"].shape[0],
            sorted_anns[0]["segmentation"].shape[1],
            4,
        )
    )
    img[:, :, 3] = 0

    for ann in sorted_anns:
        m = ann["segmentation"]
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask
        if borders:
            import cv2

            contours, _ = cv2.findContours(
                m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            # Try to smooth contours
            contours = [
                cv2.approxPolyDP(contour, epsilon=0.01, closed=True)
                for contour in contours
            ]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1)

    ax.imshow(img)


class SimpleFolderDataset(torch.utils.data.Dataset):
    def __init__(self, folder, input_list=None):
        self.folder = Path(folder)
        if input_list is None:
            image_list = [Path(f) for f in folder.glob("*.jpg")]
            print(image_list)
        else:
            # load image names from the list
            with open(input_list, "r") as f:
                image_list = [
                    self.folder / Path(line.split()[0]).name for line in f.readlines()
                ]
        self.files = image_list

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        return path.name, np.array(Image.open(path).convert("RGB"))


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Save masks for all images in a folder"
    )
    parser.add_argument(
        "--input_folder", type=str, help="Path to the folder with images"
    )
    parser.add_argument(
        "--input_list", type=str, help="Path to the file with the list of input images"
    )
    parser.add_argument(
        "--output_folder", type=str, help="Path to the folder to save masks"
    )
    # Model checkpoint and config file
    parser.add_argument(
        "--model_cfg", type=str, help="Path to the model config file", required=True
    )
    parser.add_argument(
        "--checkpoint", type=str, help="Path to the model checkpoint", required=True
    )
    parser.add_argument("--show_masks", action="store_true", help="Show masks")
    parser.add_argument(
        "--hparam_search", action="store_true", help="Hyperparameter search"
    )

    args = parser.parse_args()
    # Create output folder
    input_folder = Path(args.input_folder)
    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    dataset = SimpleFolderDataset(input_folder, input_list=args.input_list)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True
    )

    print("Loading SAM model")
    print("Model config:", args.model_cfg)
    print("Model checkpoint:", args.checkpoint)
    sam = sam_model_registry[args.model_cfg](checkpoint=args.checkpoint).to(device)
    mask_generators_dict = {}
    output_folders_dict = {}
    panoptic_jsons_dict = {}
    if args.hparam_search:
        points_per_side_list = [32, 64, 96]
        pred_iou_thresh_list = [0.5, 0.6, 0.7]
        stability_score_thresh_list = [0.9, 0.95, 0.99]
        for i, (points_per_side, pred_iou_thresh, stability_score_thresh) in enumerate(
            product(
                points_per_side_list, pred_iou_thresh_list, stability_score_thresh_list
            )
        ):
            mask_generators_dict[i] = SamAutomaticMaskGenerator(
                model=sam,
                points_per_batch=1024,
                points_per_side=points_per_side,
                pred_iou_thresh=pred_iou_thresh,
                stability_score_thresh=stability_score_thresh,
            )
            output_folder_i = Path(
                output_folder
                / f"points_per_side_{points_per_side}_pred_iou_thresh_{pred_iou_thresh}_stability_score_thresh_{stability_score_thresh}"
            )
            output_folder_i.mkdir(parents=True, exist_ok=True)
            output_folders_dict[i] = output_folder_i
            panoptic_jsons_dict[i] = {
                "images": [],
                "annotations": [],
                "categories": [
                    {
                        "id": 1,
                        "name": "visual_object",
                    }
                ],
            }
            output_folder_panoptic_id = output_folder_i / "panoptic_id"
            output_folder_panoptic_id.mkdir(parents=True, exist_ok=True)
            output_folder_panoptic_color = output_folder_i / "panoptic_color"
            output_folder_panoptic_color.mkdir(parents=True, exist_ok=True)
            output_folder_instance_anns = output_folder_i / "instance_anns"
            output_folder_instance_anns.mkdir(parents=True, exist_ok=True)

        num_models = len(mask_generators_dict)
    else:
        points_per_side = 32
        pred_iou_thresh = 0.88
        stability_score_thresh = 0.95

        mask_generators_dict[0] = SamAutomaticMaskGenerator(
            model=sam,
            points_per_batch=256,
            points_per_side=points_per_side,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
        )

        output_folder_i = Path(
            output_folder
            / f"points_per_side_{points_per_side}_pred_iou_thresh_{pred_iou_thresh}_stability_score_thresh_{stability_score_thresh}"
        )
        output_folders_dict[0] = output_folder_i
        output_folder_panoptic_id = output_folder_i / "panoptic_id"
        output_folder_panoptic_id.mkdir(parents=True, exist_ok=True)
        output_folder_panoptic_color = output_folder_i / "panoptic_color"
        output_folder_panoptic_color.mkdir(parents=True, exist_ok=True)
        output_folder_instance_anns = output_folder_i / "instance_anns"
        output_folder_instance_anns.mkdir(parents=True, exist_ok=True)

        panoptic_jsons_dict[0] = {
            "images": [],
            "annotations": [],
            "categories": [
                {
                    "id": 1,
                    "name": "visual_object",
                }
            ],
        }
        num_models = 1

    for file_name, image in tqdm.tqdm(dataloader):
        file_name = file_name[0]
        image = image[0].numpy()
        if file_name.endswith(".png") or file_name.endswith(".jpg"):
            print("Processing", file_name)
            file = input_folder / file_name
            print(f"Loading image from {file}")
            # Load image
            image_path = Path(file)
            image_id = image_path.stem
            H, W = image.shape[:2]
            # Predict masks
            print(f"Predicting masks with {num_models} models")
            for model_idx in tqdm.tqdm(range(num_models)):
                with torch.inference_mode(), torch.autocast(
                    "cuda", dtype=torch.float32
                ):
                    masks = mask_generators_dict[model_idx].generate(image)
                instance_anns = {
                    "image": {
                        "image_id": image_id,
                        "file_name": image_path.name,
                        "height": H,
                        "width": W,
                    },
                    "annotations": [],
                }
                for i, mask in enumerate(masks):
                    mask_rle_encoding = mask_utils.encode(mask["segmentation"])
                    instance_anns["annotations"].append(
                        {
                            "id": i + 1,
                            "segmentation": {
                                "size": mask_rle_encoding["size"],
                                "counts": mask_rle_encoding["counts"].decode("utf-8"),
                            },
                            "area": mask["area"],
                            "predicted_iou": mask["predicted_iou"],
                            "stability_score": mask["stability_score"],
                            "crop_box": mask["crop_box"],
                            "point_coords": mask["point_coords"],
                        }
                    )
                with open(
                    output_folders_dict[model_idx]
                    / "instance_anns"
                    / f"{image_id}.json",
                    "w",
                ) as f:
                    json.dump(instance_anns, f)

                if len(masks) == 0:
                    print("No masks found for", file)
                    continue
                if args.show_masks:
                    plt.figure(figsize=(20, 20))
                    plt.imshow(image)
                    show_anns(masks)
                    plt.axis("off")
                    plt.show()
                # Save masks
                sorted_masks = sorted(masks, key=(lambda x: x["area"]), reverse=True)

                panoptic_img = np.zeros((H, W, 3))
                panoptic_img_color = np.zeros((H, W, 3))
                initial_panoptic_img_id = np.zeros((H, W))
                segments_info = []
                id = 1
                for mask_dict in sorted_masks:
                    initial_panoptic_img_id[mask_dict["segmentation"]] = id
                    mask_dict["id"] = id
                    id += 1

                # save_counter = 0
                for mask_dict in sorted_masks:
                    mask = initial_panoptic_img_id == mask_dict["id"]
                    area = np.sum(mask)
                    if area < 512:
                        continue
                    panoptic_img[mask, :] = id2rgb(mask_dict["id"])
                    panoptic_img_color[mask, :] = np.random.randint(0, 255, 3)
                    # Image.fromarray(panoptic_img_color.astype(np.uint8)).save(f"/home/josip/temp/{image_id}_{save_counter}.png")
                    # save_counter += 1

                    mask_x, mask_y = np.where(mask)
                    bbox_center_x = int((mask_x.min() + mask_x.max()) / 2)
                    bbox_center_y = int((mask_y.min() + mask_y.max()) / 2)
                    bbox_width = int(mask_x.max() - mask_x.min())
                    bbox_height = int(mask_y.max() - mask_y.min())

                    segments_info.append(
                        {
                            "id": mask_dict["id"],
                            "category_id": 1,
                            "isthing": 1,
                            "area": int(area),
                            "bbox": [
                                bbox_center_x,
                                bbox_center_y,
                                bbox_width,
                                bbox_height,
                            ],
                            "iscrowd": 0,
                        }
                    )

                panoptic_jsons_dict[model_idx]["images"].append(
                    {
                        "id": image_id,
                        "file_name": image_path.name,
                        "width": image.shape[1],
                        "height": image.shape[0],
                    }
                )
                panoptic_jsons_dict[model_idx]["annotations"].append(
                    {
                        "image_id": image_id,
                        "file_name": f"{image_id}.png",
                        "segments_info": segments_info,
                    }
                )

                # Save panoptic image
                panoptic_img = panoptic_img.astype(np.uint8)
                panoptic_img_color = panoptic_img_color.astype(np.uint8)
                panoptic_img = Image.fromarray(panoptic_img)
                panoptic_img_color = Image.fromarray(panoptic_img_color)
                panoptic_img.save(
                    output_folders_dict[model_idx] / "panoptic_id" / f"{image_id}.png"
                )
                panoptic_img_color.save(
                    output_folders_dict[model_idx]
                    / "panoptic_color"
                    / f"{image_id}.png"
                )

    for model_idx in range(num_models):
        with open(output_folders_dict[model_idx] / "panoptic.json", "w") as f:
            print(
                "Saving panoptic json to",
                output_folders_dict[model_idx] / "panoptic.json",
            )
            json.dump(panoptic_jsons_dict[model_idx], f)


if __name__ == "__main__":
    main()
