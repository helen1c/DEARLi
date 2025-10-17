"""
# Code modified from FC-CLIP. (https://github.com/bytedance/fc-clip/blob/main/fcclip/modeling/backbone/clip.py)
"""

from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.utils.memory import retry_if_cuda_oom

from mask2former.modeling.criterion_unlabeled import SetCriterionPseudolabels

from .modeling.criterion import SetCriterion
from .modeling.matcher import HungarianMatcher
from .utils.misc import VILD_PROMPT

from .modeling.transformer_decoder.vision_language_mask2former_transformer_decoder import (
    MaskPooling,
    get_classification_logits,
)

import logging

logger = logging.getLogger(__name__)


@META_ARCH_REGISTRY.register()
class VisionLanguageMaskFormerStudent(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        criterion_labeled: nn.Module,
        criterion_pseudolabeled: nn.Module,
        num_queries: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        metadata,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        # inference
        semantic_on: bool,
        panoptic_on: bool,
        test_topk_per_image: int,
        filter_small_objects_threshold: float,
        # Vision language parameters
        geometric_ensemble_alpha: float,
        ensemble_on_valid_mask: bool,
        dim_latent: None,
        use_zero_shot_branch: True,
    ):
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.criterion_labeled = criterion_labeled
        self.criterion_pseudolabeled = criterion_pseudolabeled
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.metadata = metadata
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer(
            "pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False
        )
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        self.use_zero_shot_branch = use_zero_shot_branch
        if not self.use_zero_shot_branch:
            logger.warning(
                "STUDENT: Zero-shot branch is disabled. Please enable it if you want to use zero-shot branch."
            )
        self.semantic_on = semantic_on
        self.panoptic_on = panoptic_on
        self.test_topk_per_image = test_topk_per_image
        self.filter_small_objects_threshold = filter_small_objects_threshold

        if not self.semantic_on:
            assert self.sem_seg_postprocess_before_inference

        self.mask_pooling = MaskPooling()
        self.geometric_ensemble_alpha = geometric_ensemble_alpha
        self.ensemble_on_valid_mask = ensemble_on_valid_mask

        self.text_classifier = None

        try:
            self.void_embedding = nn.Embedding(
                1, backbone.dim_latent
            )  # use this for void
        except AttributeError:
            assert dim_latent is not None
            self.void_embedding = nn.Embedding(1, dim_latent)

        self.num_templates, self.class_names = self.prepare_class_names_from_metadata(
            metadata
        )

        assert self.sem_seg_head.num_classes == len(self.metadata.stuff_classes)

    def prepare_class_names_from_metadata(self, metadata):
        def split_labels(x):
            res = []
            for x_ in x:
                x_ = x_.replace(", ", ",")
                x_ = x_.split(",")  # there can be multiple synonyms for single class
                res.append(x_)
            return res

        def fill_all_templates_ensemble(x_=""):
            res = []
            for x in x_:
                for template in VILD_PROMPT:
                    res.append(template.format(x))
            return res, len(res) // len(VILD_PROMPT)

        # get text classifier
        try:
            class_names = split_labels(
                metadata.stuff_classes
            )  # it includes both thing and stuff
        except:
            # if only things classes
            class_names = split_labels(metadata.thing_classes)

        num_templates = []
        templated_class_names = []
        for x in class_names:
            templated_classes, templated_classes_num = fill_all_templates_ensemble(x)
            templated_class_names += templated_classes
            num_templates.append(templated_classes_num)
        class_names = templated_class_names
        return num_templates, class_names

    def get_text_classifier(self):
        if self.text_classifier is None:
            text_classifier = []
            bs = 128
            for idx in range(0, len(self.class_names), bs):
                text_classifier.append(
                    self.backbone.get_text_classifier(
                        self.class_names[idx : idx + bs], self.device
                    ).detach()
                )
            text_classifier = torch.cat(text_classifier, dim=0)
            text_classifier /= text_classifier.norm(dim=-1, keepdim=True)
            text_classifier = text_classifier.reshape(
                text_classifier.shape[0] // len(VILD_PROMPT),
                len(VILD_PROMPT),
                text_classifier.shape[-1],
            ).mean(1)
            text_classifier /= text_classifier.norm(dim=-1, keepdim=True)
            self.text_classifier = text_classifier
        return self.text_classifier, self.num_templates

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

        # Loss parameters:
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

        # loss weights
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT

        deep_supervision_pseudolabeled = (
            cfg.MODEL.STUDENT.UNSUPERVISED_LOSS.DEEP_SUPERVISION
        )
        no_object_weight_pseudolabeled = (
            cfg.MODEL.STUDENT.UNSUPERVISED_LOSS.NO_OBJECT_WEIGHT
        )

        # loss weights
        class_weight_pseudolabeled = cfg.MODEL.STUDENT.UNSUPERVISED_LOSS.CLASS_WEIGHT
        dice_weight_pseudolabeled = cfg.MODEL.STUDENT.UNSUPERVISED_LOSS.DICE_WEIGHT
        mask_weight_pseudolabeled = cfg.MODEL.STUDENT.UNSUPERVISED_LOSS.MASK_WEIGHT

        matcher_labeled = HungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )

        matcher_pseudolabeled = HungarianMatcher(
            cost_class=class_weight_pseudolabeled,
            cost_mask=mask_weight_pseudolabeled,
            cost_dice=dice_weight_pseudolabeled,
            num_points=cfg.MODEL.STUDENT.UNSUPERVISED_LOSS.TRAIN_NUM_POINTS,
        )

        weight_dict_labeled = {
            "loss_ce": class_weight,
            "loss_mask": mask_weight,
            "loss_dice": dice_weight,
        }

        weight_dict_pseudolabeled = {
            "pseudo_loss_ce": class_weight_pseudolabeled,
            "pseudo_loss_mask": mask_weight_pseudolabeled,
            "pseudo_loss_dice": dice_weight_pseudolabeled,
        }

        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update(
                    {k + f"_{i}": v for k, v in weight_dict_labeled.items()}
                )
            weight_dict_labeled.update(aux_weight_dict)

        if deep_supervision_pseudolabeled:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            exclude_in_deep_supervision = (
                cfg.MODEL.STUDENT.UNSUPERVISED_LOSS.EXCLUDE_IN_DEEP_SUPERVISION
            )
            if exclude_in_deep_supervision is None:
                exclude_in_deep_supervision = 1
            else:
                assert (
                    exclude_in_deep_supervision < dec_layers
                ), "number of layers to exclude from deep supervision must be less than number of decoder layers"
            aux_weight_dict = {}
            for i in range(exclude_in_deep_supervision - 1, dec_layers - 1):
                aux_weight_dict.update(
                    {k + f"_{i}": v for k, v in weight_dict_pseudolabeled.items()}
                )
            weight_dict_pseudolabeled.update(aux_weight_dict)

        criterion_labeled = SetCriterion(
            sem_seg_head.num_classes,
            matcher=matcher_labeled,
            weight_dict=weight_dict_labeled,
            eos_coef=no_object_weight,
            losses=cfg.MODEL.STUDENT.LOSSES_LABELED,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
        )

        criterion_pseudolabeled = SetCriterionPseudolabels(
            sem_seg_head.num_classes,
            matcher=matcher_pseudolabeled,
            weight_dict=weight_dict_pseudolabeled,
            eos_coef=no_object_weight_pseudolabeled,
            losses=cfg.MODEL.STUDENT.LOSSES_PSEUDOLABELED,
            num_points=cfg.MODEL.STUDENT.UNSUPERVISED_LOSS.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.STUDENT.UNSUPERVISED_LOSS.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.STUDENT.UNSUPERVISED_LOSS.IMPORTANCE_SAMPLE_RATIO,
        )

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion_labeled": criterion_labeled,
            "criterion_pseudolabeled": criterion_pseudolabeled,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": (
                cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                or cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON
            ),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            # inference
            "semantic_on": cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON,
            "panoptic_on": cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON,
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            "filter_small_objects_threshold": cfg.MODEL.TEACHER.FILTER_SMALL_OBJECTS_AREA_THRESHOLD,
            "geometric_ensemble_alpha": cfg.MODEL.VL_MASK_FORMER.GEOMETRIC_ENSEMBLE_ALPHA,
            "ensemble_on_valid_mask": cfg.MODEL.VL_MASK_FORMER.ENSEMBLE_ON_VALID_MASK,
            "dim_latent": cfg.MODEL.VL_MASK_FORMER.EMBED_DIM,
            "use_zero_shot_branch": cfg.MODEL.VL_MASK_FORMER.STUDENT.USE_ZERO_SHOT_BRANCH,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        images = [x["image"].to(self.device) for x in batched_inputs]
        text_classifier, num_templates = self.get_text_classifier()
        # Append void class weight
        text_classifier = torch.cat(
            [text_classifier, F.normalize(self.void_embedding.weight, dim=-1)], dim=0
        )
        if "student_image" in batched_inputs[0]:
            assert self.training, "student_image is only used in training"
            images += [x["student_image"].to(self.device) for x in batched_inputs]
            images = [(x - self.pixel_mean) / self.pixel_std for x in images]
            images_labeled = ImageList.from_tensors(
                images[: len(images) // 2], self.size_divisibility
            )

            features_labeled = self.backbone(images_labeled.tensor)
            features_labeled["text_classifier"] = text_classifier
            features_labeled["num_templates"] = num_templates
            outputs_labeled = self.sem_seg_head(features_labeled)

            images_pseudolabeled = ImageList.from_tensors(
                images[len(images) // 2 :], self.size_divisibility
            )
            features_pseudolabeled = self.backbone(images_pseudolabeled.tensor)
            features_pseudolabeled["text_classifier"] = text_classifier
            features_pseudolabeled["num_templates"] = num_templates
            outputs_pseudolabeled = self.sem_seg_head(features_pseudolabeled)
        else:
            images = [(x - self.pixel_mean) / self.pixel_std for x in images]
            images = ImageList.from_tensors(images, self.size_divisibility)
            features = self.backbone(images.tensor)
            features["text_classifier"] = text_classifier
            features["num_templates"] = num_templates
            outputs = self.sem_seg_head(features)

        if self.training:
            # mask classification target
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                targets_labeled = self.prepare_targets(gt_instances, images_labeled)
            else:
                targets_labeled = None

            if "teacher_instances_panseg" in batched_inputs[0]:
                pseudo_gt_instances = [
                    x["teacher_instances_panseg"].to(self.device)
                    for x in batched_inputs
                ]
                targets_pseudolabeled = self.prepare_pseudolabeled_targets_panoptic(
                    pseudo_gt_instances, images_pseudolabeled
                )
            elif "teacher_instances_semseg" in batched_inputs[0]:
                pseudo_gt_instances = [
                    x["teacher_instances_semseg"].to(self.device)
                    for x in batched_inputs
                ]
                targets_pseudolabeled = self.prepare_pseudolabeled_targets_semantic(
                    pseudo_gt_instances, images_pseudolabeled
                )
            else:
                targets_pseudolabeled = None

            # bipartite matching-based loss
            losses_labeled = self.criterion_labeled(outputs_labeled, targets_labeled)
            losses_pseudolabeled = self.criterion_pseudolabeled(
                outputs_pseudolabeled, targets_pseudolabeled
            )
            for k in list(losses_labeled.keys()):
                if k in self.criterion_labeled.weight_dict:
                    losses_labeled[k] *= self.criterion_labeled.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses_labeled.pop(k)
            for k in list(losses_pseudolabeled.keys()):
                if k in self.criterion_pseudolabeled.weight_dict:
                    losses_pseudolabeled[k] *= self.criterion_pseudolabeled.weight_dict[
                        k
                    ]
                else:
                    losses_pseudolabeled.pop(k)

            losses_labeled.update(losses_pseudolabeled)
            return losses_labeled

        else:
            if self.use_zero_shot_branch:

                mask_cls_results = outputs["pred_logits"]
                mask_pred_results = outputs["pred_masks"]

                # We ensemble the pred logits of in-vocab and out-vocab
                clip_feature = features["clip_vis_dense"]
                mask_for_pooling = F.interpolate(
                    mask_pred_results,
                    size=clip_feature.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                if (
                    "convnext" in self.backbone.model_name.lower()
                    or "vit" in self.backbone.model_name.lower()
                ):
                    pooled_clip_feature = self.mask_pooling(
                        clip_feature, mask_for_pooling
                    )
                    pooled_clip_feature = self.backbone.visual_prediction_forward(
                        pooled_clip_feature
                    )
                elif "rn" in self.backbone.model_name.lower():
                    pooled_clip_feature = self.backbone.visual_prediction_forward(
                        clip_feature, mask_for_pooling
                    )
                else:
                    raise NotImplementedError("Backbone not supported")

                out_vocab_cls_results = get_classification_logits(
                    pooled_clip_feature,
                    text_classifier,
                    self.backbone.clip_model.logit_scale,
                    num_templates,
                )

                in_vocab_cls_results = mask_cls_results[..., :-1]  # remove void
                out_vocab_cls_results = out_vocab_cls_results[..., :-1]  # remove void

                # Reference: https://github.com/NVlabs/ODISE/blob/main/odise/modeling/meta_arch/odise.py#L1506
                out_vocab_cls_probs = out_vocab_cls_results.softmax(-1)
                in_vocab_cls_results = in_vocab_cls_results.softmax(-1)

                if self.ensemble_on_valid_mask:
                    # Only include out_vocab cls results on masks with valid pixels
                    # We empirically find that this is important to obtain reasonable AP/mIOU score with ResNet CLIP models
                    valid_masking = (mask_for_pooling > 0).to(mask_for_pooling).sum(
                        -1
                    ).sum(-1) > 0
                    valid_masking = valid_masking.to(
                        in_vocab_cls_results.dtype
                    ).unsqueeze(-1)
                    alpha = (
                        torch.ones_like(in_vocab_cls_results)
                        * self.geometric_ensemble_alpha
                    )
                    alpha = alpha * valid_masking
                else:
                    alpha = self.geometric_ensemble_alpha

                cls_results = (
                    in_vocab_cls_results ** (1 - alpha) * out_vocab_cls_probs**alpha
                ).log()

                # This is used to filtering void predictions.
                is_void_prob = F.softmax(mask_cls_results, dim=-1)[..., -1:]
                mask_cls_probs = torch.cat(
                    [cls_results.softmax(-1) * (1.0 - is_void_prob), is_void_prob],
                    dim=-1,
                )
                mask_cls_results = torch.log(mask_cls_probs + 1e-8)

                # upsample masks
                mask_pred_results = F.interpolate(
                    mask_pred_results,
                    size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                    mode="bilinear",
                    align_corners=False,
                )

                del outputs

            else:
                mask_cls_results = outputs["pred_logits"]
                mask_pred_results = outputs["pred_masks"]
                # upsample masks
                mask_pred_results = F.interpolate(
                    mask_pred_results,
                    size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                    mode="bilinear",
                    align_corners=False,
                )

                del outputs

            processed_results = []
            for mask_cls_result, mask_pred_result, input_per_image, image_size in zip(
                mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                processed_results.append({})

                if self.sem_seg_postprocess_before_inference:
                    mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                        mask_pred_result, image_size, height, width
                    )
                    mask_cls_result = mask_cls_result.to(mask_pred_result)

                # semantic segmentation inference
                if self.semantic_on:
                    r = retry_if_cuda_oom(self.semantic_inference)(
                        mask_cls_result, mask_pred_result
                    )
                    if not self.sem_seg_postprocess_before_inference:
                        r = retry_if_cuda_oom(sem_seg_postprocess)(
                            r, image_size, height, width
                        )
                    processed_results[-1]["sem_seg"] = r

                # panoptic segmentation inference
                if self.panoptic_on:
                    panoptic_r = retry_if_cuda_oom(self.panoptic_inference)(
                        mask_cls_result, mask_pred_result
                    )
                    processed_results[-1]["panoptic_seg"] = panoptic_r

            return processed_results

    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            # pad gt
            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros(
                (gt_masks.shape[0], h_pad, w_pad),
                dtype=gt_masks.dtype,
                device=gt_masks.device,
            )
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            new_targets.append(
                {
                    "labels": targets_per_image.gt_classes,
                    "masks": padded_masks,
                }
            )
        return new_targets

    def prepare_pseudolabeled_targets_panoptic(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            # pad gt
            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros(
                (gt_masks.shape[0], h_pad, w_pad),
                dtype=gt_masks.dtype,
                device=gt_masks.device,
            )
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks

            indices_remain_objects = (
                (gt_masks.sum((-2, -1)) > self.filter_small_objects_threshold)
                .nonzero()
                .squeeze(-1)
            )
            new_targets.append(
                {
                    "labels": targets_per_image.gt_classes[indices_remain_objects],
                    "masks": padded_masks[indices_remain_objects],
                    "segment_confidence_maps": targets_per_image.segment_confidence_maps[
                        indices_remain_objects
                    ],
                    "segment_abs_mask_logits": torch.abs(
                        targets_per_image.segment_per_pixel_logits[
                            indices_remain_objects
                        ]
                    ),
                    "segment_per_pixel_logits": targets_per_image.segment_per_pixel_logits[
                        indices_remain_objects
                    ],
                }
            )
        return new_targets

    def prepare_pseudolabeled_targets_semantic(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            # pad gt
            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros(
                (gt_masks.shape[0], h_pad, w_pad),
                dtype=gt_masks.dtype,
                device=gt_masks.device,
            )
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            new_targets.append(
                {
                    "labels": targets_per_image.gt_classes,
                    "masks": padded_masks,
                    # "segment_confidence_maps": targets_per_image.segment_confidence_maps,
                }
            )
        return new_targets

    def semantic_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg

    def panoptic_inference(self, mask_cls, mask_pred):
        scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
        mask_pred = mask_pred.sigmoid()
        num_classes = len(self.metadata.stuff_classes)
        keep = labels.ne(num_classes) & (scores > self.object_mask_threshold)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        cur_mask_cls = mask_cls[keep]
        cur_mask_cls = cur_mask_cls[:, :-1]

        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
        segments_info = []

        current_segment_id = 0

        if cur_masks.shape[0] == 0:
            # We didn't detect any mask :(
            return panoptic_seg, segments_info
        else:
            # take argmax
            cur_mask_ids = cur_prob_masks.argmax(0)
            stuff_memory_list = {}
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                isthing = (
                    pred_class
                    in self.metadata.thing_dataset_id_to_contiguous_id.values()
                )
                mask_area = (cur_mask_ids == k).sum().item()
                original_area = (cur_masks[k] >= 0.5).sum().item()
                mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

                if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                    if mask_area / original_area < self.overlap_threshold:
                        continue

                    # merge stuff regions
                    if not isthing:
                        if int(pred_class) in stuff_memory_list.keys():
                            panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                            continue
                        else:
                            stuff_memory_list[int(pred_class)] = current_segment_id + 1

                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id

                    segments_info.append(
                        {
                            "id": current_segment_id,
                            "isthing": bool(isthing),
                            "category_id": int(pred_class),
                        }
                    )

            return panoptic_seg, segments_info
