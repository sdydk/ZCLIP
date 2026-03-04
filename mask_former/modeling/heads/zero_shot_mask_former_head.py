# Copyright (c) Facebook, Inc. and its affiliates.
import logging
from copy import deepcopy
from typing import Callable, Dict, List, Optional, Tuple, Union

import fvcore.nn.weight_init as weight_init
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY

from ..transformer.zero_shot_transformer_predictor import ZeroShotTransformerPredictor
from .pixel_decoder import build_pixel_decoder


@SEM_SEG_HEADS_REGISTRY.register()
class ZeroShotMaskFormerHead(nn.Module):

    _version = 2

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)
        if version is None or version < 2:
            # Do not warn if train from scratch
            scratch = True
            logger = logging.getLogger(__name__)
            for k in list(state_dict.keys()):
                newk = k
                if "sem_seg_head" in k and not k.startswith(prefix + "predictor"):
                    newk = k.replace(prefix, prefix + "pixel_decoder.")
                    # logger.debug(f"{k} ==> {newk}")
                if newk != k:
                    state_dict[newk] = state_dict[k]
                    del state_dict[k]
                    scratch = False

            if not scratch:
                logger.warning(
                    f"Weight format of {self.__class__.__name__} have changed! "
                    "Please upgrade your models. Applying automatic conversion now ..."
                )

    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        num_classes: int,
        pixel_decoder: nn.Module,
        loss_weight: float = 1.0,
        ignore_value: int = -1,
        # extra parameters
        transformer_predictor: nn.Module,
        transformer_in_feature: str,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            num_classes: number of classes to predict
            pixel_decoder: the pixel decoder module
            loss_weight: loss weight
            ignore_value: category id to be ignored during training.
            transformer_predictor: the transformer decoder that makes prediction
            transformer_in_feature: input feature name to the transformer_predictor
        """
        super().__init__()
        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        self.in_features = [k for k, v in input_shape]
        feature_strides = [v.stride for k, v in input_shape]
        feature_channels = [v.channels for k, v in input_shape]

        self.ignore_value = ignore_value
        self.common_stride = 4
        self.loss_weight = loss_weight

        self.pixel_decoder = pixel_decoder
        self.predictor = transformer_predictor
        self.transformer_in_feature = transformer_in_feature

        self.num_classes = num_classes

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        return {
            "input_shape": {
                k: v
                for k, v in input_shape.items()
                if k in cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES
            },
            "ignore_value": cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
            "num_classes": cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            "pixel_decoder": build_pixel_decoder(cfg, input_shape),
            "loss_weight": cfg.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT,
            "transformer_in_feature": cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE,
            "transformer_predictor": ZeroShotTransformerPredictor(
                cfg,
                cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
                if cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE == "transformer_encoder"
                else input_shape[cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE].channels,
                mask_classification=True,
            ),
        }
    # 2026-03-03 text_layers_features, images_layer_features, images_vision_shape, image_txt_feature 这些变量都是新添加的为了是将CLIP提取的图像特征和文本特征融到分割掩码中
    def forward(self, features, text_layers_features, images_layer_features, images_vision_shape, image_txt_feature): #  
        return self.layers(features, text_layers_features, images_layer_features, images_vision_shape, image_txt_feature) # 

    def layers(self, features, text_layers_features, images_layer_features, images_vision_shape, image_txt_feature):
        # 2026-03-03 多输入进去一个image_txt_feature  它是CLIP文本编码器提取到的最后一层输出 transformer_encoder_features是None
        (
            mask_features,
            transformer_encoder_features,
        ) = self.pixel_decoder.forward_features(features, image_txt_feature)
        
        if self.transformer_in_feature == "transformer_encoder":
            assert (
                transformer_encoder_features is not None
            ), "Please use the TransformerEncoderPixelDecoder."
            predictions = self.predictor(transformer_encoder_features, mask_features)
        else:
            # 2026-03-03 features[self.transformer_in_feature]---res5 多了 text_layers_features, images_layer_features, images_vision_shape  
            # 2026-03-03 Transformer Predictor的输出是out,其中肯定包含"pred_masks" = outputs_seg_masks
            # outputs_seg_masks来源: mask_embed = MLP(hs), outputs_seg_masks = torch.einsum("lbqc,bchw->lbqhw", mask_embed, mask_features), mask_features表示的Pixel decoder的输出

            predictions = self.predictor(
                features[self.transformer_in_feature], mask_features, text_layers_features, images_layer_features, images_vision_shape
            )
        return predictions

    def freeze_pretrained(self):
        for name, module in self.named_children():
            if name not in ["predictor"]:
                for param in module.parameters():
                    param.requires_grad = False
            else:
                module.freeze_pretrained()
