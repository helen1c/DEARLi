from detectron2.engine import SimpleTrainer
import time
import torch
from detectron2.structures import Instances, BitMasks
from mask2former.utils.misc import fuse_segment_per_pixel_logits


class SemiSupTrainRunner(SimpleTrainer):
    def __init__(
        self,
        teacher,
        student,
        data_loader,
        optimizer,
        gather_metric_period=1,
        zero_grad_before_forward=False,
        async_write_metrics=False,
    ):
        super().__init__(
            student,
            data_loader,
            optimizer,
            gather_metric_period=gather_metric_period,
            zero_grad_before_forward=zero_grad_before_forward,
            async_write_metrics=async_write_metrics,
        )
        self.teacher = teacher

    def obtain_cutmixed_image(self, batch, cutmix_box_mask):
        student_image_cutmixed = batch["student_image"].clone()
        student_image_cutmixed[:, cutmix_box_mask] = batch["cutmix_addons"][
            "student_image_cutmix"
        ][:, cutmix_box_mask]

        return student_image_cutmixed

    def obtain_cutmixed_panseg(self, batch, cutmix_box_mask):
        teacher_panseg = batch["teacher_pan_seg"]
        teacher_panseg_cutmix = batch["teacher_pan_seg_cutmix"].clone()
        max_id = teacher_panseg.max()
        teacher_panseg_cutmix[teacher_panseg_cutmix > 0] += max_id
        teacher_panseg[cutmix_box_mask] = teacher_panseg_cutmix[cutmix_box_mask]

        classes = []
        masks = []
        segment_confidence_maps = []
        segment_per_pixel_logits_list = []

        unique_ids = torch.unique(teacher_panseg)
        stuff_classes_map = {}
        for ind, segment_info in enumerate(batch["teacher_segments_info"]):
            assert ind == segment_info["id"] - 1
            if segment_info["id"] not in unique_ids:
                continue
            cat_id = segment_info["category_id"]
            if segment_info["isthing"]:
                classes.append(cat_id)
                masks.append(teacher_panseg == segment_info["id"])
                segment_confidence_maps.append(
                    batch["teacher_instances_panseg"].segment_confidence_maps[
                        segment_info["id"] - 1
                    ]
                )
                segment_per_pixel_logits_list.append(
                    batch["teacher_instances_panseg"].segment_per_pixel_logits[
                        segment_info["id"] - 1
                    ]
                )
            else:
                if cat_id not in stuff_classes_map:
                    stuff_classes_map[cat_id] = len(classes)
                    classes.append(cat_id)
                    masks.append(teacher_panseg == segment_info["id"])
                    segment_confidence_maps.append(
                        batch["teacher_instances_panseg"].segment_confidence_maps[
                            segment_info["id"] - 1
                        ]
                    )
                    segment_per_pixel_logits_list.append(
                        batch["teacher_instances_panseg"].segment_per_pixel_logits[
                            segment_info["id"] - 1
                        ]
                    )
                else:
                    masks[stuff_classes_map[cat_id]] |= (
                        teacher_panseg == segment_info["id"]
                    )
                    segment_confidence_maps[stuff_classes_map[cat_id]] = torch.max(
                        segment_confidence_maps[stuff_classes_map[cat_id]],
                        batch["teacher_instances_panseg"].segment_confidence_maps[
                            segment_info["id"] - 1
                        ],
                    )
                    segment_per_pixel_logits_list[stuff_classes_map[cat_id]] = (
                        fuse_segment_per_pixel_logits(
                            segment_per_pixel_logits_list[stuff_classes_map[cat_id]],
                            batch["teacher_instances_panseg"].segment_per_pixel_logits[
                                segment_info["id"] - 1
                            ],
                        )
                    )

        for ind, segment_info in enumerate(batch["teacher_segments_info_cutmix"]):
            assert ind == segment_info["id"] - 1
            if segment_info["id"] + max_id not in unique_ids:
                continue
            cat_id = segment_info["category_id"]
            new_segment_id = segment_info["id"] + max_id
            if segment_info["isthing"]:
                classes.append(cat_id)
                masks.append(teacher_panseg == (new_segment_id))
                segment_confidence_maps.append(
                    batch["teacher_instances_panseg_cutmix"].segment_confidence_maps[
                        segment_info["id"] - 1
                    ]
                )
                segment_per_pixel_logits_list.append(
                    batch["teacher_instances_panseg_cutmix"].segment_per_pixel_logits[
                        segment_info["id"] - 1
                    ]
                )
            else:
                if cat_id not in stuff_classes_map:
                    stuff_classes_map[cat_id] = len(classes)
                    classes.append(cat_id)
                    masks.append(teacher_panseg == (new_segment_id))
                    segment_confidence_maps.append(
                        batch[
                            "teacher_instances_panseg_cutmix"
                        ].segment_confidence_maps[segment_info["id"] - 1]
                    )
                    segment_per_pixel_logits_list.append(
                        batch[
                            "teacher_instances_panseg_cutmix"
                        ].segment_per_pixel_logits[segment_info["id"] - 1]
                    )
                else:
                    masks[stuff_classes_map[cat_id]] |= teacher_panseg == (
                        new_segment_id
                    )
                    segment_confidence_maps[stuff_classes_map[cat_id]] = torch.max(
                        segment_confidence_maps[stuff_classes_map[cat_id]],
                        batch[
                            "teacher_instances_panseg_cutmix"
                        ].segment_confidence_maps[segment_info["id"] - 1],
                    )
                    segment_per_pixel_logits_list[stuff_classes_map[cat_id]] = (
                        fuse_segment_per_pixel_logits(
                            segment_per_pixel_logits_list[stuff_classes_map[cat_id]],
                            batch[
                                "teacher_instances_panseg_cutmix"
                            ].segment_per_pixel_logits[segment_info["id"] - 1],
                        )
                    )

        new_instances = Instances((teacher_panseg.shape[0], teacher_panseg.shape[1]))
        new_instances.gt_classes = torch.tensor(classes, dtype=torch.int64)
        if len(masks) == 0:
            new_instances.gt_masks = torch.zeros(
                (0, teacher_panseg.shape[0], teacher_panseg.shape[1])
            )
            new_instances.segment_confidence_maps = torch.zeros(
                (0, teacher_panseg.shape[0], teacher_panseg.shape[1])
            )
            new_instances.segment_per_pixel_logits = torch.zeros(
                (0, teacher_panseg.shape[0], teacher_panseg.shape[1])
            )
        else:
            masks = BitMasks(torch.stack(masks))
            new_instances.gt_masks = masks.tensor
            segment_confidence_maps = torch.stack(segment_confidence_maps)
            new_instances.segment_confidence_maps = segment_confidence_maps
            segment_per_pixel_logits = torch.stack(segment_per_pixel_logits_list)
            new_instances.segment_per_pixel_logits = segment_per_pixel_logits
        return new_instances

    def run_step(self):
        """
        Implement the standard training logic described above.
        """
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        """
        If you want to do something with the data, you can wrap the dataloader.
        """
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        if self.zero_grad_before_forward:
            """
            If you need to accumulate gradients or do something similar, you can
            wrap the optimizer with your custom `zero_grad()` method.
            """
            self.optimizer.zero_grad()

        """
        If you want to do something with the losses, you can wrap the model.
        """

        with torch.no_grad():
            self.teacher.eval()

            cutmix_indices = [i for i in range(len(data)) if "cutmix_addons" in data[i]]
            if len(cutmix_indices) > 0:
                data_cutmixed = [data[i] for i in cutmix_indices]
                teacher_outputs_cutmix = self.teacher(data_cutmixed, cutmix=True)
            teacher_outputs = self.teacher(data)
        for i, (batch, teacher_output) in enumerate(zip(data, teacher_outputs)):
            if self.teacher.panoptic_on:
                batch["teacher_instances_panseg"] = teacher_output["panoptic_seg"][
                    "panoptic_instances"
                ]
                if teacher_output.get("decoder_embeddings", None) is not None:
                    batch["teacher_decoder_embeddings"] = teacher_output[
                        "decoder_embeddings"
                    ]
                if "cutmix_addons" in batch:
                    if batch["cutmix_addons"]["cutmix_box"].sum() > 0:
                        batch["teacher_pan_seg"] = teacher_output["panoptic_seg"][
                            "panoptic_seg"
                        ]
                        batch["teacher_segments_info"] = teacher_output["panoptic_seg"][
                            "segments_info"
                        ]
                        teacher_outputs_index = cutmix_indices.index(i)
                        batch["teacher_instances_panseg_cutmix"] = (
                            teacher_outputs_cutmix[teacher_outputs_index][
                                "panoptic_seg"
                            ]["panoptic_instances"]
                        )
                        batch["teacher_pan_seg_cutmix"] = teacher_outputs_cutmix[
                            teacher_outputs_index
                        ]["panoptic_seg"]["panoptic_seg"]
                        batch["teacher_segments_info_cutmix"] = teacher_outputs_cutmix[
                            teacher_outputs_index
                        ]["panoptic_seg"]["segments_info"]
                        # prepare cutmixed image
                        cutmix_box_mask = batch["cutmix_addons"]["cutmix_box"] == 1
                        batch["student_image"] = self.obtain_cutmixed_image(
                            batch=batch, cutmix_box_mask=cutmix_box_mask
                        )
                        batch["teacher_instances_panseg"] = self.obtain_cutmixed_panseg(
                            batch=batch, cutmix_box_mask=cutmix_box_mask
                        )
            else:
                raise NotImplementedError(
                    "Currently we only support panoptic segmentation for semi-supervised learning."
                )

        loss_dict = self.model(data)
        if isinstance(loss_dict, torch.Tensor):
            losses = loss_dict
            loss_dict = {"total_loss": loss_dict}
        else:
            losses = sum(loss_dict.values())
        if not self.zero_grad_before_forward:
            """
            If you need to accumulate gradients or do something similar, you can
            wrap the optimizer with your custom `zero_grad()` method.
            """
            self.optimizer.zero_grad()
        losses.backward()
        self.after_backward()

        if self.async_write_metrics:
            # write metrics asynchronically
            self.concurrent_executor.submit(
                self._write_metrics, loss_dict, data_time, iter=self.iter
            )
        else:
            self._write_metrics(loss_dict, data_time)
        self.optimizer.step()
