"""
# Code modified from FC-CLIP. (https://github.com/bytedance/fc-clip/blob/main/fcclip/modeling/backbone/clip.py)
"""

import logging
import math
from typing import List, Optional

import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec
from detectron2.utils import comm
from timm.layers import trunc_normal_

logger = logging.getLogger(__name__)


@BACKBONE_REGISTRY.register()
class OpenCLIPBackbone(Backbone):
    def __init__(self, cfg, input_shape):
        super().__init__()
        model_name = cfg.MODEL.VL_MASK_FORMER.OPEN_CLIP_MODEL_NAME
        pretrained = cfg.MODEL.VL_MASK_FORMER.OPEN_CLIP_PRETRAINED_WEIGHTS
        logger.info(f"Using {model_name} with pretrained weights {pretrained}.")

        # download on local rank 0 first
        if comm.get_local_rank() == 0:
            open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        comm.synchronize()

        self.model_name = model_name
        self.pretrained = pretrained

        self.clip_model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )

        self.freeze_backbone = cfg.MODEL.VL_MASK_FORMER.FREEZE_BACKBONE

        self._out_feature_strides = {
            "stem": 2,
            "res2": 4,
            "res3": 8,
            "res4": 16,
            "res5": 32,
            "clip_embedding": -1,
        }

        self.text_tokenizer = open_clip.get_tokenizer(model_name)
        self.output_channels = self.resolve_output_channels(model_name)
        self.dim_latent = self.clip_model.text_projection.shape[-1]

        self._out_feature_channels = {
            "stem": self.output_channels[0],
            "res2": self.output_channels[1],
            "res3": self.output_channels[2],
            "res4": self.output_channels[3],
            "res5": self.output_channels[4],
            "clip_embedding": self.dim_latent,
        }

        self.extract_features_methods = {
            "convnext": self.extract_features_convnext,
        }

        self.visual_prediction_forward_methods = {
            "convnext": self.visual_prediction_forward_convnext,
        }

        if self.freeze_backbone:
            self.eval()
            self.freeze_everything()
        else:
            self.freeze_everything_except_feature_extractor()

    def freeze_everything_except_feature_extractor(self):
        for name, param in self.clip_model.named_parameters():
            if (
                "transformer.resblocks" in name
                or "visual.head" in name
                or "visual.trunk.head" in name
                or "logit_scale" in name
                or "text_projection" in name
                or "positional_embedding" in name
                or "ln_final" in name
                or "token_embedding" in name
            ):
                param.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def resolve_output_channels(self, model_name):
        model_name = model_name.lower()
        if "convnext_" in model_name:
            self.model_type = "convnext"
            if "_base" in model_name:
                return [128, 128, 256, 512, 1024]
            elif "_large" in model_name:
                return [192, 192, 384, 768, 1536]
            elif "_xxlarge" in model_name:
                return [384, 384, 768, 1536, 3072]
        else:
            raise NotImplementedError(
                f"Model {model_name} not supported yet. You can add your implementation here."
            )

    def freeze_everything(self):
        for param in self.clip_model.parameters():
            param.requires_grad = False

    def encode_text(self, text, normalize: bool = False):
        return self.clip_model.encode_text(text, normalize=normalize)

    def tokenize_text(self, text):
        return self.text_tokenizer(text)

    def extract_features(self, x):
        return self.extract_features_methods[self.model_type](x)

    def visual_prediction_forward(self, x, masks=None):
        return self.visual_prediction_forward_methods[self.model_type](x, masks)

    def extract_features_convnext(self, x):
        out = {}
        x = self.clip_model.visual.trunk.stem(x)
        out["stem"] = x.contiguous()  # os4
        for i in range(4):
            x = self.clip_model.visual.trunk.stages[i](x)
            out[f"res{i+2}"] = (
                x.contiguous()
            )  # res 2 (os4), 3 (os8), 4 (os16), 5 (os32)

        x = self.clip_model.visual.trunk.norm_pre(x)
        out["clip_vis_dense"] = x.contiguous()
        return out

    def visual_prediction_forward_convnext(self, x, masks):
        batch, num_query, channel = x.shape
        x = x.reshape(batch * num_query, channel, 1, 1)  # fake 2D input
        x = self.clip_model.visual.trunk.head(x)
        x = self.clip_model.visual.head(x)
        return x.view(batch, num_query, x.shape[-1])  # B x num_queries x 640

    def get_text_classifier(self, text_list, device):
        self.eval()
        with torch.no_grad():
            text_tokens = self.tokenize_text(text_list)
            text_tokens = text_tokens.to(device)
            # we return un-normalized text feature.
            text_features = self.encode_text(text_tokens, normalize=False)
            return text_features

    def forward(self, x):
        if self.freeze_backbone:
            self.eval()
            with torch.no_grad():
                return self.extract_features(x)
        else:
            return self.extract_features(x)

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name],
                stride=self._out_feature_strides[name],
            )
            for name in ["stem", "res2", "res3", "res4", "res5", "clip_embedding"]
        }

    @property
    def size_divisibility(self):
        return -1
