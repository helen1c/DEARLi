# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import logging

import numpy as np
import torch
from detectron2.utils import comm
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import detection_utils as utils, DatasetCatalog
from detectron2.data import transforms as T
from detectron2.structures import BitMasks, Instances

from .mask_former_semantic_dataset_mapper import MaskFormerSemanticDatasetMapper
import torchvision.transforms.v2 as tv_tv2
from random import choice
import PIL
import matplotlib.pyplot as plt
import random
from PIL import ImageFilter

logger = logging.getLogger(__name__)

__all__ = ["MaskFormerPanopticSemiSupDatasetMapper"]


def obtain_cutmix_box(
    img_size, p=0.5, size_min=0.02, size_max=0.4, ratio_1=0.3, ratio_2=1 / 0.3
):
    mask = torch.zeros(img_size[0], img_size[1])
    if random.random() > p:
        return mask

    size = np.random.uniform(size_min, size_max) * img_size[0] * img_size[1]
    while True:
        ratio = np.random.uniform(ratio_1, ratio_2)
        cutmix_w = int(np.sqrt(size / ratio))
        cutmix_h = int(np.sqrt(size * ratio))
        x = np.random.randint(0, img_size[1])
        y = np.random.randint(0, img_size[0])

        if x + cutmix_w <= img_size[1] and y + cutmix_h <= img_size[0]:
            break

    mask[y : y + cutmix_h, x : x + cutmix_w] = 1

    return mask


class GaussianBlur:
    """
    Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709
    Adapted from MoCo:
    https://github.com/facebookresearch/moco/blob/master/moco/loader.py
    Note that this implementation does not seem to be exactly the same as
    described in SimCLR.
    """

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class MaskFormerPanopticSemiSupDatasetMapper(MaskFormerSemanticDatasetMapper):
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by MaskFormer for panoptic segmentation.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    @configurable
    def __init__(
        self,
        is_train=True,
        *,
        augmentations,
        image_format,
        ignore_label,
        size_divisibility,
        unlabeled_dataset,
        labeled_dataset,
        unlabeled_transform,
        student_jitter,
        cutmix_enabled,
        cutmix_prob,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            ignore_label: the label that is ignored to evaluation
            size_divisibility: pad image size to be divisible by this value
        """
        super().__init__(
            is_train,
            augmentations=augmentations,
            image_format=image_format,
            ignore_label=ignore_label,
            size_divisibility=size_divisibility,
        )
        self.unlabeled_dataset = unlabeled_dataset
        self.labeled_dataset = labeled_dataset
        self.unlabeled_transform = unlabeled_transform
        self.student_jitter = student_jitter
        self.cutmix_enabled = cutmix_enabled
        self.cutmix_prob = cutmix_prob
        print(f"CutMix is set to {self.cutmix_enabled}, with prob {self.cutmix_prob}.")

    @classmethod
    def from_config(cls, cfg, is_train=True):
        ret = super().from_config(cfg, is_train)
        unlabeled_dataset = DatasetCatalog.get(cfg.DATASETS.TRAIN_UNLABELED)
        assert (
            len(cfg.DATASETS.TRAIN) == 1
        ), "Currently, only one labeled dataset is supported"
        labeled_dataset = DatasetCatalog.get(cfg.DATASETS.TRAIN[0])
        ret["unlabeled_dataset"] = unlabeled_dataset
        ret["labeled_dataset"] = labeled_dataset
        augs = [
            T.ResizeShortestEdge(
                cfg.UNLABELED_INPUT.MIN_SIZE_TRAIN,
                cfg.UNLABELED_INPUT.MAX_SIZE_TRAIN,
                cfg.UNLABELED_INPUT.MIN_SIZE_TRAIN_SAMPLING,
            )
        ]
        if cfg.UNLABELED_INPUT.CROP.ENABLED:
            augs.append(
                T.RandomCrop_CategoryAreaConstraint(
                    cfg.UNLABELED_INPUT.CROP.TYPE,
                    cfg.UNLABELED_INPUT.CROP.SIZE,
                    cfg.UNLABELED_INPUT.CROP.SINGLE_CATEGORY_MAX_AREA,
                    cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                )
            )
        augs.append(T.RandomFlip())
        ret["unlabeled_transform"] = augs
        student_transform = []
        if cfg.UNLABELED_INPUT.STUDENT_JITTER.COLOR_JITTER.ENABLED:
            student_transform.append(
                tv_tv2.ColorJitter(
                    brightness=cfg.UNLABELED_INPUT.STUDENT_JITTER.COLOR_JITTER.BRIGHTNESS,
                    contrast=cfg.UNLABELED_INPUT.STUDENT_JITTER.COLOR_JITTER.CONTRAST,
                    saturation=cfg.UNLABELED_INPUT.STUDENT_JITTER.COLOR_JITTER.SATURATION,
                    hue=cfg.UNLABELED_INPUT.STUDENT_JITTER.COLOR_JITTER.HUE,
                )
            )
            student_transform.append(
                tv_tv2.RandomGrayscale(
                    p=cfg.UNLABELED_INPUT.STUDENT_JITTER.RANDOM_GRAYSCALE_PROB
                )
            )
            student_transform.append(
                tv_tv2.RandomApply(
                    [
                        GaussianBlur(
                            cfg.UNLABELED_INPUT.STUDENT_JITTER.GAUSSIAN_BLUR_SIGMA
                        )
                    ],
                    p=cfg.UNLABELED_INPUT.STUDENT_JITTER.GAUSSIAN_BLUR_PROB,
                )
            )

        ret["student_jitter"] = tv_tv2.Compose(student_transform)
        ret["cutmix_enabled"] = cfg.UNLABELED_INPUT.CUTMIX_ENABLED
        ret["cutmix_prob"] = cfg.UNLABELED_INPUT.CUTMIX_PROB
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        assert (
            self.is_train
        ), "MaskFormerPanopticDatasetMapper should only be used for training!"

        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        unlabeled_image_path = choice(self.unlabeled_dataset)["file_name"]
        unlabeled_image = utils.read_image(unlabeled_image_path, format=self.img_format)

        # semantic segmentation
        if "sem_seg_file_name" in dataset_dict:
            # PyTorch transformation not implemented for uint16, so converting it to double first
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name")).astype(
                "double"
            )
        else:
            sem_seg_gt = None

        # panoptic segmentation
        if "pan_seg_file_name" in dataset_dict:
            pan_seg_gt = utils.read_image(dataset_dict.pop("pan_seg_file_name"), "RGB")
            segments_info = dataset_dict["segments_info"]
        else:
            pan_seg_gt = None
            segments_info = None

        if pan_seg_gt is None:
            raise ValueError(
                "Cannot find 'pan_seg_file_name' for panoptic segmentation dataset {}.".format(
                    dataset_dict["file_name"]
                )
            )

        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
        aug_input, transforms = T.apply_transform_gens(self.tfm_gens, aug_input)
        image = aug_input.image
        if sem_seg_gt is not None:
            sem_seg_gt = aug_input.sem_seg

        unsup_aug_input = T.AugInput(unlabeled_image)
        unsup_aug_input, unsup_transforms = T.apply_transform_gens(
            self.unlabeled_transform, unsup_aug_input
        )
        unlabeled_image = unsup_aug_input.image
        teacher_image = unlabeled_image
        student_image = self.transform_student_image(copy.deepcopy(unlabeled_image))
        pan_seg_gt = transforms.apply_segmentation(pan_seg_gt)

        from panopticapi.utils import rgb2id

        pan_seg_gt = rgb2id(pan_seg_gt)

        # Pad image and segmentation label here!
        image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        teacher_image = torch.as_tensor(
            np.ascontiguousarray(teacher_image.transpose(2, 0, 1))
        )
        student_image = torch.as_tensor(
            np.ascontiguousarray(student_image.transpose(2, 0, 1))
        )

        if sem_seg_gt is not None:
            sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))
        pan_seg_gt = torch.as_tensor(pan_seg_gt.astype("long"))

        if self.cutmix_enabled:
            unlabeled_image_path_cutmix = choice(self.unlabeled_dataset)["file_name"]
            unlabeled_image_cutmix = utils.read_image(
                unlabeled_image_path_cutmix, format=self.img_format
            )
            unsup_aug_input_cutmix = T.AugInput(unlabeled_image_cutmix)
            (
                unsup_aug_input_cutmix,
                unsup_transforms_cutmix,
            ) = T.apply_transform_gens(self.unlabeled_transform, unsup_aug_input_cutmix)
            unlabeled_image_cutmix = unsup_aug_input_cutmix.image
            teacher_image_cutmix = unlabeled_image_cutmix
            student_image_cutmix = self.transform_student_image(
                copy.deepcopy(unlabeled_image_cutmix)
            )
            teacher_image_cutmix = torch.as_tensor(
                np.ascontiguousarray(teacher_image_cutmix.transpose(2, 0, 1))
            )
            cutmix_box = obtain_cutmix_box(
                teacher_image_cutmix.shape[-2:], p=self.cutmix_prob
            )
            student_image_cutmix = torch.as_tensor(
                np.ascontiguousarray(student_image_cutmix.transpose(2, 0, 1))
            )

        if self.size_divisibility > 0:
            image_size = (image.shape[-2], image.shape[-1])
            padding_size = [
                0,
                self.size_divisibility - image_size[1],
                0,
                self.size_divisibility - image_size[0],
            ]
            image = F.pad(image, padding_size, value=128).contiguous()
            if sem_seg_gt is not None:
                sem_seg_gt = F.pad(
                    sem_seg_gt, padding_size, value=self.ignore_label
                ).contiguous()
            pan_seg_gt = F.pad(
                pan_seg_gt, padding_size, value=0
            ).contiguous()  # 0 is the VOID panoptic label

            unlabeled_image_size = (teacher_image.shape[-2], teacher_image.shape[-1])

            padding_size = [
                0,
                self.size_divisibility - unlabeled_image_size[1],
                0,
                self.size_divisibility - unlabeled_image_size[0],
            ]
            teacher_image = F.pad(teacher_image, padding_size, value=128).contiguous()
            student_image = F.pad(student_image, padding_size, value=128).contiguous()

            if self.cutmix_enabled:
                unlabeled_image_size = (
                    teacher_image_cutmix.shape[-2],
                    teacher_image_cutmix.shape[-1],
                )

                padding_size = [
                    0,
                    self.size_divisibility - unlabeled_image_size[1],
                    0,
                    self.size_divisibility - unlabeled_image_size[0],
                ]
                teacher_image_cutmix = F.pad(
                    teacher_image_cutmix, padding_size, value=128
                ).contiguous()
                student_image_cutmix = F.pad(
                    student_image_cutmix, padding_size, value=128
                ).contiguous()
                cutmix_box = F.pad(cutmix_box, padding_size, value=0).contiguous()

        image_shape = (image.shape[-2], image.shape[-1])  # h, w

        dataset_dict["image"] = image
        dataset_dict["teacher_image"] = teacher_image
        dataset_dict["student_image"] = student_image

        if self.cutmix_enabled and cutmix_box.sum() > 0:
            dataset_dict["cutmix_addons"] = {
                "teacher_image_cutmix": teacher_image_cutmix,
                "student_image_cutmix": student_image_cutmix,
                "cutmix_box": cutmix_box,
            }

        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = sem_seg_gt.long()

        if "annotations" in dataset_dict:
            raise ValueError(
                "Semantic segmentation dataset should not have 'annotations'."
            )

        # Prepare per-category binary masks
        pan_seg_gt = pan_seg_gt.numpy()
        instances = Instances(image_shape)
        classes = []
        masks = []
        for segment_info in segments_info:
            if not segment_info["iscrowd"]:
                class_id = segment_info["category_id"]
                mask = pan_seg_gt == segment_info["id"]
                if mask.sum() > 0:
                    classes.append(class_id)
                    masks.append(mask)

        classes = np.array(classes)

        instances.gt_classes = torch.tensor(classes, dtype=torch.int64)

        if len(masks) == 0:
            # Some image does not have annotation (all ignored)
            instances.gt_masks = torch.zeros(
                (0, pan_seg_gt.shape[-2], pan_seg_gt.shape[-1])
            )
        else:
            masks = BitMasks(
                torch.stack(
                    [torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks]
                )
            )
            instances.gt_masks = masks.tensor

        dataset_dict["instances"] = instances
        return dataset_dict

    def transform_student_image(self, student_image):
        if self.img_format == "BGR":
            student_image = student_image[:, :, ::-1]
        student_image = self.student_jitter(PIL.Image.fromarray(student_image))
        student_image = np.array(student_image)
        if self.img_format == "BGR":
            student_image = student_image[:, :, ::-1]
        return student_image
