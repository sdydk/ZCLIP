# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/transformer.py
"""
Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from .position_encoding import PositionEmbeddingSine


class Transformer(nn.Module):
    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        return_intermediate_dec=False,
    ):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before
        )
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm
        )

        decoder_layer = TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before
        )
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            decoder_norm,
            return_intermediate=return_intermediate_dec,
        )

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    # 2026-03-03 src [2, 2048, 16, 16]--->[2, 256, 16, 16] mask是None images_layer_tokens [12, 2, 768, 32, 32] 多了text_layers_features, images_layer_tokens
    # 2026-03-03 pos_embed表示的是经过位置编码操(self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True))的Res5
    # 2026-03-03 query_embed = self.query_embed = nn.Embedding(num_queries, hidden_dim)
    def forward(self, src, mask, query_embed, pos_embed, text_layers_features, images_layer_tokens):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape 
        src = src.flatten(2).permute(2, 0, 1) # [2, 256, 16, 16]--->[2, 256, 256(16*16)]--->[256(16*16), 2, 256]
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1) 
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        if mask is not None:
            mask = mask.flatten(1)
        # # 2026-03-03 按照src的处理方式进行维度处理对CLIP提取的图像特征进行处理
        # images_layer_tokens = images_layer_tokens.flatten(3).permute(0, 3, 1, 2)
        
        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        # 2026-03-03 多了text_layers_features和images_layer_tokens
        hs = self.decoder(
            tgt,
            memory,
            memory_key_padding_mask=mask,
            pos=pos_embed,
            query_pos=query_embed,
            text_layers_features=text_layers_features,
            images_layer_tokens=images_layer_tokens
        )
        return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        src,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        output = src

        for layer in self.layers:
            output = layer(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                pos=pos,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(
            q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(
            q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)



class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        text_layers_features : Optional[Tensor] = None,
        images_layer_tokens : Optional[Tensor] = None
    ):
        # 2026-03-03 tgt是一个全零矩阵,其维度(N queries [100, 512])跟query一样  [100, 2, 256]
        # 2026-03-03 memory表示的Res5经过TransformerEncoder处理过后的特征[256, 2, 256]   [1024, 2, 768]
        output = tgt  

        intermediate = []
        index = 0
        for layer in self.layers:
            
            # # 2026-03-03 此处的layer_feature和vis_layer_feature是我们新添加的
            # layer_feature = text_layers_features.permute(1, 0, 2)[index, :, :]
            # vis_layer_feature = images_layer_tokens[index, ...]  #2026-01-11 是一层层的交互，还是一块拼接处理？？？？ 
            layer_feature=None
            vis_layer_feature=None
            
            output = layer(
                output, 
                memory, 
                tgt_mask=tgt_mask, # None
                memory_mask=memory_mask, # None
                tgt_key_padding_mask=tgt_key_padding_mask, # None
                memory_key_padding_mask=memory_key_padding_mask, # None
                pos=pos,
                query_pos=query_pos,
                layer_feature=layer_feature,
                vis_layer_feature=vis_layer_feature,
            )
            index = index + 1
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)



class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        # # 2026-03-03 下边的操作是利用交叉注意力将视觉特征融合到TransformerDecoder的每一层中
        # self.norm4 = nn.LayerNorm(d_model)
        # self.dropout4 = nn.Dropout(dropout)
        # self.cross_dot_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # self.vis_proj = nn.Linear(768, 256)
        # nn.init.normal_(self.vis_proj.weight, std=768 ** -0.5)
        # self.vis_proj = self.vis_proj.to("cuda")

        # # self.text_proj = nn.Linear(512, 256) # 对双曲文本特征进行维度变换，从512变成256，方便跟视觉特征进行交叉注意力计算
        # # nn.init.normal_(self.text_proj.weight, std=512 ** -0.5)
        # # self.text_proj = self.text_proj.to("cuda")

        # # self.pe_layer_v = PositionEmbeddingSine(128, normalize=True)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    # 2026-03-03 layer_feature和vis_layer_feature是新添加的 # [50, 512]
    def forward_post(
        self,
        tgt, # [100, 2, 256]
        memory, # [256, 2, 256]
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        layer_feature: Optional[Tensor] = None, 
        vis_layer_feature: Optional[Tensor] = None,
    ):  
        # 2026-01-09应该做的事情 如何将双曲文本特征与视觉特征进行交叉注意力
        # print("================tgt1", tgt.shape, memory.shape, vis_layer_feature.shape)  # 现在需要将vis_layer_feature维度（[1024(32*32), 2, 768]）进行修改
        # print("================tgt2", vis_layer_feature.shape) 
        
        # 2026-03-03 给tgt=torch.zeros_like(query_embed)添加位置编码  query_pos==self.query_embed = nn.Embedding(num_queries, hidden_dim) 加上位置编码
        # 2026-03-03 self.with_pos_embed是位置编码操作: [100, 2, 256]---2张图像，每张图像100个掩码，每个掩码的特征向量维度是256
        q = k = self.with_pos_embed(tgt, query_pos) 
        tgt2 = self.self_attn(
            q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos), 
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        # # 2026-01-09:实现了双曲文本特征与视觉特征的交叉注意力计算 .detach() 切断layer_feature与原CLIP双曲映射计算图的关联，仅保留张量值，避免反向传播时访问已经释放的计算图  
        # # 2026-01-10: 多卡训练时添加了.detach()会导致梯度无法进行更新   
        # layer_feature = self.text_proj(layer_feature)
        # proj_text_feat = layer_feature.unsqueeze(1).repeat(1, tgt.shape[1], 1) #   proj_text_feat = proj_text_feat.clone()  # 克隆张量，保留梯度追踪但不复用原计算图
        # tgt3 = self.cross_dot_attn(
        #     query=self.with_pos_embed(tgt, query_pos),
        #     key=proj_text_feat,
        #     value=proj_text_feat,
        #     attn_mask=None,
        #     key_padding_mask=None,
        # )[0] # 文本特征与视觉特征进行交叉注意力计算：注入语义先验知识，实现文本指导的分割，对齐图文特征（解决视觉-文本鸿沟）；让分割查询token可以关联到上层类别，提升细粒度/长尾类别的分割精度
        # tgt = tgt + self.dropout3(tgt3)
        # tgt = self.norm3(tgt)

        # # 2026-03-03 17:40 将视觉特征融进来时进行的一些操作，不用文本特征了
        # vis_layer_feature = self.vis_proj(vis_layer_feature)
        # # print("==========vis_layer_feature", vis_layer_feature.shape)  # vis_layer_feature 应该是 [1024, 2, 256]【2， 256， 32，32】 
        # # pos_v = self.pe_layer_v(vis_layer_feature)
        # # 训练([1024, 2, 256]) torch.Size([100, 2, 256])
        # # 测试不一样 ([1024, 2, 256]) torch.Size([100, 1, 256])
        # # print("=============vis_layer_feature", vis_layer_feature.shape, tgt.shape)
        # tgt3 = self.cross_dot_attn(
        #     query=self.with_pos_embed(tgt, query_pos),
        #     key=vis_layer_feature,
        #     value=vis_layer_feature,
        #     attn_mask=None,
        #     key_padding_mask=None,
        # )[0] # 文本特征与视觉特征进行交叉注意力计算：注入语义先验知识，实现文本指导的分割，对齐图文特征（解决视觉-文本鸿沟）；让分割查询token可以关联到上层类别，提升细粒度/长尾类别的分割精度
        # tgt = tgt + self.dropout3(tgt3)
        # tgt = self.norm3(tgt) #[100, 2, 256]

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(
            q, k, value=tgt2, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt2, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        layer_feature: Optional[Tensor] = None,
        vis_layer_feature: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(
                tgt,
                memory,
                tgt_mask,
                memory_mask,
                tgt_key_padding_mask,
                memory_key_padding_mask,
                pos,
                query_pos,
            )
        
        return self.forward_post(
            tgt,
            memory,
            tgt_mask,
            memory_mask,
            tgt_key_padding_mask,
            memory_key_padding_mask,
            pos,
            query_pos,
            layer_feature,
            vis_layer_feature
        )




def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")
