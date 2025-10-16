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

from .modeling.criterion import SetCriterion
from .modeling.matcher import HungarianMatcher
from .utils.misc import VILD_PROMPT, fuse_segment_per_pixel_logits
from copy import deepcopy, copy
from .modeling.transformer_decoder.vision_language_mask2former_transformer_decoder import (
    MaskPooling,
    get_classification_logits,
)
from torch.nn.parallel import DistributedDataParallel
import logging
import math

logger = logging.getLogger(__name__)


@META_ARCH_REGISTRY.register()
class VisionLanguageMaskFormerTeacher(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    def __init__(
        self,
        cfg,
        maskformer,
        ema_decay,
        burn_in_iters,
        panoptic_on=True,
        object_mask_threshold=None,
        alpha_statistic_logger=None,
    ):
        super().__init__()
        self.ema_decay = ema_decay
        self.burn_in_iters = burn_in_iters
        self.teacher_weights_path_set = cfg.MODEL.TEACHER.WEIGHTS.strip() != ""
        self.init_modules(maskformer, use_deepcopy=True)
        self.convert_weights_to_leaf_nodes()

        self.num_queries = maskformer.num_queries
        self.overlap_threshold = maskformer.overlap_threshold
        if object_mask_threshold is None:
            self.object_mask_threshold = maskformer.object_mask_threshold
        else:
            self.object_mask_threshold = object_mask_threshold

        print(f"Using {self.object_mask_threshold} threshold in teacher.")

        self.metadata = maskformer.metadata
        self.size_divisibility = maskformer.size_divisibility
        self.sem_seg_postprocess_before_inference = (
            maskformer.sem_seg_postprocess_before_inference
        )
        self.register_buffer(
            "pixel_mean", torch.Tensor(maskformer.pixel_mean).view(-1, 1, 1), False
        )
        self.register_buffer(
            "pixel_std", torch.Tensor(maskformer.pixel_std).view(-1, 1, 1), False
        )
        self.alpha_statistic_logger = alpha_statistic_logger
        self.alpha_scheduler = None
        self.panoptic_on = panoptic_on

        self.test_topk_per_image = maskformer.test_topk_per_image
        assert self.sem_seg_postprocess_before_inference

        self.mask_pooling = MaskPooling()
        self.geometric_ensemble_alpha = maskformer.geometric_ensemble_alpha
        self.ensemble_on_valid_mask = maskformer.ensemble_on_valid_mask

        self.text_classifier = maskformer.text_classifier

        self.num_templates, self.class_names = (
            maskformer.num_templates,
            maskformer.class_names,
        )
        self.use_zero_shot_branch = (
            cfg.MODEL.VL_MASK_FORMER.TEACHER.USE_ZERO_SHOT_BRANCH
        )
        if not self.use_zero_shot_branch:
            logger.warning(
                "TEACHER: Zero-shot branch is disabled. Please enable it if you want to use zero-shot branch."
            )

        assert len(self.metadata.stuff_classes) == self.sem_seg_head.num_classes

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

    def init_modules(self, maskformer_model, use_deepcopy):
        if use_deepcopy:
            self.backbone = deepcopy(maskformer_model.backbone)
            self.sem_seg_head = deepcopy(maskformer_model.sem_seg_head)
            self.void_embedding = deepcopy(maskformer_model.void_embedding)
        else:
            self.backbone = copy(maskformer_model.backbone)
            self.sem_seg_head = copy(maskformer_model.sem_seg_head)
            self.void_embedding = copy(maskformer_model.void_embedding)

    def convert_weights_to_leaf_nodes(self):
        for name, param in self.named_parameters():
            param.detach()
            param.requires_grad = False

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs, cutmix=False):
        if cutmix:
            images = [
                x["cutmix_addons"]["teacher_image_cutmix"].to(self.device)
                for x in batched_inputs
            ]
        else:
            images = [x["teacher_image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        text_classifier, num_templates = self.get_text_classifier()
        text_classifier = torch.cat(
            [text_classifier, F.normalize(self.void_embedding.weight, dim=-1)], dim=0
        )

        features = self.backbone(images.tensor)
        # Append void class weight
        features["text_classifier"] = text_classifier
        features["num_templates"] = num_templates
        outputs = self.sem_seg_head(features)

        if self.training:
            raise NotImplementedError("MaskFormerTeacher is only used for inference.")
        else:

            mask_cls_results = outputs["pred_logits"]
            mask_pred_results = outputs["pred_masks"]

            if self.use_zero_shot_branch:

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
                    raise NotImplementedError()

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
                    c_alpha = (
                        self.alpha_scheduler.get_alpha()
                        if self.alpha_scheduler is not None
                        else self.geometric_ensemble_alpha
                    )
                    alpha = torch.ones_like(in_vocab_cls_results) * c_alpha
                    alpha = alpha * valid_masking
                else:
                    alpha = (
                        self.alpha_scheduler.get_alpha()
                        if self.alpha_scheduler is not None
                        else self.geometric_ensemble_alpha
                    )

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
            processed_results = []

            for mask_cls_result, mask_pred_result, input_per_image, image_size in zip(
                mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes
            ):
                height, width = image_size
                processed_results.append({})

                if self.sem_seg_postprocess_before_inference:
                    mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                        mask_pred_result, image_size, height, width
                    )
                    mask_cls_result = mask_cls_result.to(mask_pred_result)

                # panoptic segmentation inference
                if self.panoptic_on:
                    panoptic_r = retry_if_cuda_oom(self.panoptic_inference)(
                        mask_cls_result, mask_pred_result
                    )
                    processed_results[-1]["panoptic_seg"] = panoptic_r

            return processed_results

    def panoptic_inference(self, mask_cls, mask_pred):
        scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
        mask_pred_logits = mask_pred
        mask_pred = mask_pred.sigmoid()

        keep = labels.ne(self.sem_seg_head.num_classes) & (
            scores > self.object_mask_threshold
        )
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        cur_mask_pred_logits = mask_pred_logits[keep]
        cur_mask_cls = mask_cls[keep]
        cur_mask_cls = cur_mask_cls[:, :-1]

        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
        segments_info = []
        instances = Instances((h, w))
        current_segment_id = 0

        if cur_masks.shape[0] == 0:
            # We didn't detect any mask :(
            instances.gt_masks = torch.zeros((0, h, w))
            instances.gt_classes = torch.tensor([], dtype=torch.int64)
            instances.segment_confidence_maps = torch.zeros(
                (0, h, w), dtype=torch.float32
            )
            instances.segment_per_pixel_logits = torch.zeros(
                (0, h, w), dtype=torch.float32
            )
            ret = {
                "prob_masks": cur_prob_masks,
                "panoptic_seg": panoptic_seg,
                "segments_info": segments_info,
                "panoptic_instances": instances,
            }
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
                            panoptic_seg[mask] = stuff_memory_list[int(pred_class)][0]
                            segments_info[stuff_memory_list[int(pred_class)][1]][
                                "segment_confidence_map"
                            ] = torch.max(
                                segments_info[stuff_memory_list[int(pred_class)][1]][
                                    "segment_confidence_map"
                                ],
                                cur_prob_masks[k],
                            )
                            segments_info[stuff_memory_list[int(pred_class)][1]][
                                "segment_per_pixel_logits"
                            ] = fuse_segment_per_pixel_logits(
                                segments_info[stuff_memory_list[int(pred_class)][1]][
                                    "segment_per_pixel_logits"
                                ],
                                cur_mask_pred_logits[k],
                            )
                            continue
                        else:
                            stuff_memory_list[int(pred_class)] = (
                                current_segment_id + 1,
                                len(segments_info),
                            )

                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id
                    segments_info.append(
                        {
                            "id": current_segment_id,
                            "isthing": bool(isthing),
                            "category_id": int(pred_class),
                            "segment_confidence_map": cur_prob_masks[k],
                            "segment_per_pixel_logits": cur_mask_pred_logits[k],
                        }
                    )

            classes = []
            masks = []
            segment_confidence_maps = []
            segment_per_pixel_logits = []

            for segment_info in segments_info:
                class_id = segment_info["category_id"]
                classes.append(class_id)
                masks.append(panoptic_seg == segment_info["id"])
                segment_confidence_maps.append(
                    segment_info.pop("segment_confidence_map")
                )
                segment_per_pixel_logits.append(
                    segment_info.pop("segment_per_pixel_logits")
                )

            instances.gt_classes = torch.tensor(classes, dtype=torch.int64)

            if len(masks) == 0:
                # Some image does not have annotation (all ignored)
                instances.gt_masks = torch.zeros((0, h, w))
                instances.segment_confidence_maps = torch.zeros(
                    (0, h, w), dtype=torch.float32
                )
                instances.segment_per_pixel_logits = torch.zeros(
                    (0, h, w), dtype=torch.float32
                )
            else:
                # Maybe make a deepcopy of masks tensors?
                masks = BitMasks(torch.stack(masks))
                instances.gt_masks = masks.tensor
                instances.segment_confidence_maps = torch.stack(segment_confidence_maps)
                instances.segment_per_pixel_logits = torch.stack(
                    segment_per_pixel_logits
                )
            ret = {
                "panoptic_seg": panoptic_seg,
                "segments_info": segments_info,
                "panoptic_instances": instances,
                "prob_masks": cur_prob_masks,
            }
        return ret

    def post_optim_step(self, student, iter):
        if isinstance(student, DistributedDataParallel):
            student = student.module
        # Burn in phase is finished...
        if iter > 0 and iter == self.burn_in_iters:
            # Copy weights from student to teacher
            self.init_modules(student, use_deepcopy=True)
            self.convert_weights_to_leaf_nodes()
            print("End of the burn-in phase. Copying weights from student to teacher.")
        elif iter >= self.burn_in_iters:
            ema_decay = (
                min(1 - (1 / (iter + 1)), self.ema_decay)
                if self.burn_in_iters == 0 and not self.teacher_weights_path_set
                else self.ema_decay
            )
            self.ema_update_weights(student, ema_decay=ema_decay)

    def ema_update_weights(self, student, ema_decay):
        # Update parameters.
        with torch.no_grad():
            for ema_param, param in zip(self.parameters(), student.parameters()):
                if not param.data.shape:  # scalar tensor
                    ema_param.data = (
                        ema_decay * ema_param.data + (1 - ema_decay) * param.data
                    )
                else:
                    ema_param.data[:] = (
                        ema_decay * ema_param[:].data[:]
                        + (1 - ema_decay) * param[:].data[:]
                    )

            for ema_buffer, buffer in zip(self.buffers(), student.buffers()):
                if not buffer.data.shape:
                    ema_buffer.data = (
                        ema_decay * ema_buffer.data + (1 - ema_decay) * buffer.data
                    )
                else:
                    ema_buffer.data[:] = (
                        ema_decay * ema_buffer[:].data[:]
                        + (1 - ema_decay) * buffer[:].data[:]
                    )
