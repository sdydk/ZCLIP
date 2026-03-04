# Copyright (c) Facebook, Inc. and its affiliates.
import logging
from typing import Callable, Dict, List, Optional, Tuple, Union

import fvcore.nn.weight_init as weight_init
from torch import nn
from torch.nn import functional as F
import torch
from detectron2.config import configurable
from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY

from ..transformer.position_encoding import PositionEmbeddingSine
from ..transformer.transformer import TransformerEncoder, TransformerEncoderLayer


def build_pixel_decoder(cfg, input_shape):
    """
    Build a pixel decoder from `cfg.MODEL.MASK_FORMER.PIXEL_DECODER_NAME`.
    """
    name = cfg.MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME
    model = SEM_SEG_HEADS_REGISTRY.get(name)(cfg, input_shape)
    forward_features = getattr(model, "forward_features", None)
    if not callable(forward_features):
        raise ValueError(
            "Only SEM_SEG_HEADS with forward_features method can be used as pixel decoder. "
            f"Please implement forward_features for {name} to only return mask features."
        )
    return model


@SEM_SEG_HEADS_REGISTRY.register()
class BasePixelDecoder(nn.Module):
    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        conv_dim: int,
        mask_dim: int,
        norm: Optional[Union[str, Callable]] = None,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            conv_dims: number of output channels for the intermediate conv layers.
            mask_dim: number of output channels for the final conv layer.
            norm (str or callable): normalization for all conv layers
        """
        super().__init__()

        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        self.in_features = [k for k, v in input_shape]  # starting from "res2" to "res5"
        feature_channels = [v.channels for k, v in input_shape]
        
        lateral_convs = []
        output_convs = []
        
        # # 2026-01-13 20:42 文本感知注意力机制  # both the visual input and for combining, num of channels # v_in # l_in # key # value
        # self.fusions = PWAM(dim, dim, 768, dim, dim, num_heads=num_heads_fusion, dropout=fusion_drop)
        fusion_modules = []

        use_bias = norm == ""
        for idx, in_channels in enumerate(feature_channels):
            if idx == len(self.in_features) - 1:
                output_norm = get_norm(norm, conv_dim)
                output_conv = Conv2d(
                    in_channels,
                    conv_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=use_bias,
                    norm=output_norm,
                    activation=F.relu,
                )
                
                weight_init.c2_xavier_fill(output_conv)
                self.add_module("layer_{}".format(idx + 1), output_conv)

                lateral_convs.append(None)
                output_convs.append(output_conv)

                # 2026-03-03 在解码器阶段通过PWAM模块来将文本信息融进去
                fusion_module = PWAM(in_channels, in_channels, 512, in_channels, in_channels, num_heads=8, dropout=0.1)
                self.add_module("fusion_{}".format(idx + 1), fusion_module)
                fusion_modules.append(fusion_module)

            else:
                lateral_norm = get_norm(norm, conv_dim)
                output_norm = get_norm(norm, conv_dim)

                lateral_conv = Conv2d(
                    in_channels,
                    conv_dim,
                    kernel_size=1,
                    bias=use_bias,
                    norm=lateral_norm,
                )
                output_conv = Conv2d(
                    conv_dim,
                    conv_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=use_bias,
                    norm=output_norm,
                    activation=F.relu,
                )
                weight_init.c2_xavier_fill(lateral_conv)
                weight_init.c2_xavier_fill(output_conv)
                self.add_module("adapter_{}".format(idx + 1), lateral_conv)
                self.add_module("layer_{}".format(idx + 1), output_conv)

                lateral_convs.append(lateral_conv)
                output_convs.append(output_conv)

                # 2026-03-03 此处与上述功能一样
                fusion_module = PWAM(in_channels, in_channels, 512, in_channels, in_channels, num_heads=8, dropout=0.1)
                self.add_module("fusion_{}".format(idx + 1), fusion_module)
                fusion_modules.append(fusion_module)

        # Place convs into top-down order (from low to high resolution)
        # to make the top-down computation in forward clearer.
        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]
        
        # 2026-03-03 文本信息融合模块的到置顺序与卷积层相反，因为是从高层特征到低层特征进行融合的
        self.fusion_modules = fusion_modules[::-1]

        self.mask_dim = mask_dim
        self.mask_features = Conv2d(
            conv_dim,
            mask_dim,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        weight_init.c2_xavier_fill(self.mask_features)


    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        ret = {}
        ret["input_shape"] = {
            k: v
            for k, v in input_shape.items()
            if k in cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES
        }
        ret["conv_dim"] = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        ret["norm"] = cfg.MODEL.SEM_SEG_HEAD.NORM
        return ret

    def forward_features(self, features, image_txt_feature):
        # Reverse feature maps into top-down order (from low to high resolution)
        for idx, f in enumerate(self.in_features[::-1]): # 所有阶段的特征都用到了
            
            x = features[f]
            
            # 2026-01-13 20:40 在这加一个注意力机制 让特征与文本信息进行交互
            b, c, h, w = x.shape

            x_a = x.flatten(2).transpose(1, 2)
            # print("视觉特征", f, x.shape, x_a.shape, b, c, h, w)
            # print("查看文本特征", image_txt_feature[0].shape, image_txt_feature[1].shape)
            # print("视觉特征和文本特征交叉注意力", self.fusion_modules[idx])
            x_1 = self.fusion_modules[idx](x_a, image_txt_feature, l_mask=None)  # x1 表示的是文本增强后的视觉特征
            x_1 = x_1.transpose(1, 2).view(b, c, h, w)
            # 添加一个残差连接操作
            x = x + x_1
            # print("原始视觉特征和文本增强后的视觉特征", idx, x.shape, x_1.shape)
            

            lateral_conv = self.lateral_convs[idx]
            output_conv = self.output_convs[idx]
        
            if lateral_conv is None: # Res5没有经过lateral_conv  下次试试不对Res进行特这个增强
                y = output_conv(x)
            else:
                cur_fpn = lateral_conv(x)
                # Following FPN implementation, we use nearest upsampling here 这是FPN
                y = cur_fpn + F.interpolate(y, size=cur_fpn.shape[-2:], mode="nearest")
                y = output_conv(y)
        return self.mask_features(y), None

    def forward(self, features, targets=None):
        logger = logging.getLogger(__name__)
        logger.warning(
            "Calling forward() may cause unpredicted behavior of PixelDecoder module."
        )
        return self.forward_features(features)



class TransformerEncoderOnly(nn.Module):
    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before
        )
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm
        )

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        if mask is not None:
            mask = mask.flatten(1)

        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        return memory.permute(1, 2, 0).view(bs, c, h, w)


@SEM_SEG_HEADS_REGISTRY.register()
class TransformerEncoderPixelDecoder(BasePixelDecoder):
    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        transformer_dropout: float,
        transformer_nheads: int,
        transformer_dim_feedforward: int,
        transformer_enc_layers: int,
        transformer_pre_norm: bool,
        conv_dim: int,
        mask_dim: int,
        norm: Optional[Union[str, Callable]] = None,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            transformer_dropout: dropout probability in transformer
            transformer_nheads: number of heads in transformer
            transformer_dim_feedforward: dimension of feedforward network
            transformer_enc_layers: number of transformer encoder layers
            transformer_pre_norm: whether to use pre-layernorm or not
            conv_dims: number of output channels for the intermediate conv layers.
            mask_dim: number of output channels for the final conv layer.
            norm (str or callable): normalization for all conv layers
        """
        super().__init__(input_shape, conv_dim=conv_dim, mask_dim=mask_dim, norm=norm)

        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        self.in_features = [k for k, v in input_shape]  # starting from "res2" to "res5"
        feature_strides = [v.stride for k, v in input_shape]
        feature_channels = [v.channels for k, v in input_shape]

        in_channels = feature_channels[len(self.in_features) - 1]
        self.input_proj = Conv2d(in_channels, conv_dim, kernel_size=1)
        weight_init.c2_xavier_fill(self.input_proj)
        self.transformer = TransformerEncoderOnly(
            d_model=conv_dim,
            dropout=transformer_dropout,
            nhead=transformer_nheads,
            dim_feedforward=transformer_dim_feedforward,
            num_encoder_layers=transformer_enc_layers,
            normalize_before=transformer_pre_norm,
        )
        N_steps = conv_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

        # update layer
        use_bias = norm == ""
        output_norm = get_norm(norm, conv_dim)
        output_conv = Conv2d(
            conv_dim,
            conv_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=use_bias,
            norm=output_norm,
            activation=F.relu,
        )
        weight_init.c2_xavier_fill(output_conv)
        delattr(self, "layer_{}".format(len(self.in_features)))
        self.add_module("layer_{}".format(len(self.in_features)), output_conv)
        self.output_convs[0] = output_conv

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        ret = super().from_config(cfg, input_shape)
        ret["transformer_dropout"] = cfg.MODEL.MASK_FORMER.DROPOUT
        ret["transformer_nheads"] = cfg.MODEL.MASK_FORMER.NHEADS
        ret["transformer_dim_feedforward"] = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD
        ret[
            "transformer_enc_layers"
        ] = cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS  # a separate config
        ret["transformer_pre_norm"] = cfg.MODEL.MASK_FORMER.PRE_NORM
        return ret

    def forward_features(self, features):
        # Reverse feature maps into top-down order (from low to high resolution)
        for idx, f in enumerate(self.in_features[::-1]):
            x = features[f]
            lateral_conv = self.lateral_convs[idx]
            output_conv = self.output_convs[idx]
            if lateral_conv is None:
                transformer = self.input_proj(x)
                pos = self.pe_layer(x)
                transformer = self.transformer(transformer, None, pos)
                y = output_conv(transformer)
                # save intermediate feature as input to Transformer decoder
                transformer_encoder_features = transformer
            else:
                cur_fpn = lateral_conv(x)
                # Following FPN implementation, we use nearest upsampling here
                y = cur_fpn + F.interpolate(y, size=cur_fpn.shape[-2:], mode="nearest")
                y = output_conv(y)
        return self.mask_features(y), transformer_encoder_features

    def forward(self, features, targets=None):
        logger = logging.getLogger(__name__)
        logger.warning(
            "Calling forward() may cause unpredicted behavior of PixelDecoder module."
        )
        return self.forward_features(features)



class PWAM(nn.Module):
    def __init__(self, dim, v_in_channels, l_in_channels, key_channels, value_channels, num_heads=0, dropout=0.0):
        super(PWAM, self).__init__()
        # input x shape: (B, H*W, dim)
        self.vis_project = nn.Sequential(nn.Conv1d(dim, dim, 1, 1),  # the init function sets bias to 0 if bias is True
                                         nn.GELU(),
                                         nn.Dropout(dropout)
                                        )

        self.image_lang_att = SpatialImageLanguageAttention(v_in_channels,  # v_in
                                                            l_in_channels,  # l_in
                                                            key_channels,  # key
                                                            value_channels,  # value
                                                            out_channels=value_channels,  # out
                                                            num_heads=num_heads)

        self.project_mm = nn.Sequential(nn.Conv1d(value_channels, value_channels, 1, 1),
                                        nn.GELU(),
                                        nn.Dropout(dropout)
                                        )

    def forward(self, x, l, l_mask):
        # input x shape: (B, H*W, dim)
        # print("=========x", x.shape)
        vis = self.vis_project(x.permute(0, 2, 1))  # (B, dim, H*W)
        # print("=========vis", vis.shape)

        lang = self.image_lang_att(x, l, l_mask)  # (B, H*W, dim)
        
        # print("lang表示PWAM模块中的Gi", lang.shape)
        lang = lang.permute(0, 2, 1)  # (B, dim, H*W)

        mm = torch.mul(vis, lang)  # 对应里边的点乘操作
        mm = self.project_mm(mm)  # (B, dim, H*W)  # 对应里边的Wio

        mm = mm.permute(0, 2, 1)  # (B, H*W, dim)
        # print("mm表示的是最终的输出特征Fi", mm.shape)
        return mm


class SpatialImageLanguageAttention(nn.Module):
    def __init__(self, v_in_channels, l_in_channels, key_channels, value_channels, out_channels=None, num_heads=1):
        super(SpatialImageLanguageAttention, self).__init__()
        # x shape: (B, H*W, v_in_channels)
        # l input shape: (B, l_in_channels, N_l)
        # l_mask shape: (B, N_l, 1)
        self.v_in_channels = v_in_channels
        self.l_in_channels = l_in_channels
        self.out_channels = out_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        self.num_heads = num_heads
        if out_channels is None:
            self.out_channels = self.value_channels

        # Keys: language features: (B, l_in_channels, #words)
        # avoid any form of spatial normalization because a sentence contains many padding 0s
        self.f_key = nn.Sequential(
            nn.Conv1d(self.l_in_channels, self.key_channels, kernel_size=1, stride=1),
        )

        # Queries: visual features: (B, H*W, v_in_channels)
        self.f_query = nn.Sequential(
            nn.Conv1d(self.v_in_channels, self.key_channels, kernel_size=1, stride=1),
            nn.InstanceNorm1d(self.key_channels),
        )

        # Values: language features: (B, l_in_channels, #words)
        self.f_value = nn.Sequential(
            nn.Conv1d(self.l_in_channels, self.value_channels, kernel_size=1, stride=1),
        )

        # Out projection
        self.W = nn.Sequential(
            nn.Conv1d(self.value_channels, self.out_channels, kernel_size=1, stride=1),
            nn.InstanceNorm1d(self.out_channels),
        )
        
        # # 2026-03-04 对KV加入门控机制，先不加入门控
        # self.key_gate = nn.Sequential(
        #     nn.Linear(self.value_channels, self.value_channels, bias=False),
        #     nn.ReLU(),
        #     nn.Linear(self.value_channels, self.value_channels, bias=False),
        #     nn.Tanh()
        # )

        # self.value_gate = nn.Sequential(
        #     nn.Linear(self.value_channels, self.value_channels, bias=False),
        #     nn.ReLU(),
        #     nn.Linear(self.value_channels, self.value_channels, bias=False),
        #     nn.Tanh()
        # )

    def forward(self, x, l, l_mask):
        # x shape: (B, H*W, v_in_channels)
        # l input shape: (B, l_in_channels, N_l)
        # l_mask shape: (B, N_l, 1)

        # 2026-01-13 21:55  进行维度的扩充
        max_dim0 = max(t.shape[0] for t in l)
        # print("列表中各个tensor最大的维度", max_dim0)
        # print("空间图像语言注意力1", len(l), l[0].shape, l[1].shape)

        aligned_list = []
        for i, t in enumerate(l):
            current_dim0 = t.shape[0]
            if current_dim0 < max_dim0:
                # 复制扩充：维度0重复max_dim0次，维度1不重复
                repeat_times = (max_dim0 + current_dim0 - 1) // current_dim0
                expanded_t = t.repeat(repeat_times, 1)
                aligned_t = expanded_t[:max_dim0, :]
                # aligned_t = t.repeat(max_dim0, 1)
                # print(f"📌 列表第{i+1}个Tensor：从 [{current_dim0}, 512] 扩充为 [{max_dim0}, 512]", aligned_t.shape)
            else:
                aligned_t = t
                # print(f"📌 列表第{i+1}个Tensor：维度已满足，无需扩充（{t.shape}）", aligned_t.shape)
            aligned_list.append(aligned_t)

        # print("aligned_list", aligned_list[0].shape, aligned_list[1].shape)

        concatenated_tensor = torch.cat([x.unsqueeze(0) for x in aligned_list], dim=0)  # 表示的就是文本信息，这里考虑加一个门控机制

        B, HW = x.size(0), x.size(1)
        x = x.permute(0, 2, 1)  # (B, key_channels, H*W)
        # l_mask = l_mask.permute(0, 2, 1)  # (B, N_l, 1) -> (B, 1, N_l)

        query = self.f_query(x)  # (B, key_channels, H*W) if Conv1D   这里用的是1*1卷积加标准化
        query = query.permute(0, 2, 1)  # (B, H*W, key_channels)
        
        # 文本信息当作Key和Value  
        # print("concatenated_tensor", query.shape, concatenated_tensor.shape)
        concatenated_tensor = concatenated_tensor.permute(0, 2, 1)
        key = self.f_key(concatenated_tensor)  # (B, key_channels, N_l)
        value = self.f_value(concatenated_tensor)  # (B, self.value_channels, N_l)
        # print("文本特征shape1", key.shape, value.shape)
        
        # # 2026-01-26 对KV进行门控机制  效果降低了!!!
        # key = key_r * self.key_gate(key_r.permute(0, 2, 1)).permute(0, 2, 1)
        # value = value_r * self.value_gate(value_r.permute(0, 2, 1)).permute(0, 2, 1)


        
        n_l = value.size(-1)  # 下边进行的是多头注意力机制，消融实验中可以添加这一块
        query = query.reshape(B, HW, self.num_heads, self.key_channels//self.num_heads).permute(0, 2, 1, 3)
        # (b, num_heads, H*W, self.key_channels//self.num_heads)
        key = key.reshape(B, self.num_heads, self.key_channels//self.num_heads, n_l)
        # (b, num_heads, self.key_channels//self.num_heads, n_l)
        value = value.reshape(B, self.num_heads, self.value_channels//self.num_heads, n_l)
        # # (b, num_heads, self.value_channels//self.num_heads, n_l)  
        # l_mask = l_mask.unsqueeze(1)  # (b, 1, 1, n_l)
        # print("文本特征shape2", query.shape, key.shape, value.shape)
        
        sim_map = torch.matmul(query, key)  # (B, self.num_heads, H*W, N_l)
        sim_map = (self.key_channels ** -.5) * sim_map  # scaled dot product
        # print("sim_map", sim_map.shape)
        
        # sim_map = sim_map + (1e4*l_mask - 1e4)  # assign a very small number to padding positions
        sim_map = F.softmax(sim_map, dim=-1)  # (B, num_heads, h*w, N_l)  进行Softmax操作
        
        out = torch.matmul(sim_map, value.permute(0, 1, 3, 2))  # (B, num_heads, H*W, self.value_channels//num_heads) [2, 8, 256, 256]
        out = out.permute(0, 2, 1, 3).contiguous().reshape(B, HW, self.value_channels)  # (B, H*W, value_channels) [2, 256, 2048]
        out = out.permute(0, 2, 1)  # (B, value_channels, HW)
        out = self.W(out)  # (B, value_channels, HW)  [2, 2048, 256]
        out = out.permute(0, 2, 1)  # (B, HW, value_channels)  1维卷积+标准化 [2, 256, 2048]
        return out

