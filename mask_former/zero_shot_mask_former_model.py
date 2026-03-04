# Copyright (c) Facebook, Inc. and its affiliates.
import logging
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F
import json
from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import ImageList
from detectron2.utils.logger import log_first_n
from .modeling.clip_adapter import (
    ClipAdapter,
    MaskFormerClipAdapter,
    build_prompt_learner,
)
from .mask_former_model import MaskFormer

def build_category_tensor(read_categories, category2idx):
    """
    将读取到的类别名列表转为对应索引的Tensor向量
    :param read_categories: 从TXT读取的类别名列表（如['Calanus sinicus', 'Sagitta crassa']）
    :return: 对应索引的Tensor（如tensor([1, 2])）
    """
    # 步骤1：过滤有效类别名（仅保留在固定列表中的）
    valid_indices = []
    for name in read_categories:
        if name in category2idx:
            valid_indices.append(category2idx[name])
        else:
            print(f"⚠️  警告：类别名「{name}」不在固定列表中，已跳过")
    
    # 步骤2：转为PyTorch Tensor（默认int64类型，符合深度学习常用格式）
    if valid_indices:
        category_tensor = torch.tensor(valid_indices, dtype=torch.int64)
    else:
        # 无有效类别时返回空tensor（避免报错）
        category_tensor = torch.tensor([], dtype=torch.int64)
    
    return category_tensor

@META_ARCH_REGISTRY.register()
class ZeroShotMaskFormer(MaskFormer):
    """
    Main class for zero shot mask classification semantic segmentation architectures.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        clip_adapter: nn.Module,
        region_clip_adapter: nn.Module = None,
        criterion: nn.Module,
        num_queries: int,
        panoptic_on: bool,
        object_mask_threshold: float,
        overlap_threshold: float,
        metadata,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        clip_ensemble: bool,
        clip_ensemble_weight: float,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            clip_adapter: adapter for clip-based mask classification
            num_queries: int, number of queries
            panoptic_on: bool, whether to output panoptic segmentation prediction
            object_mask_threshold: float, threshold to filter query based on classification score
                for panoptic segmentation inference
            overlap_threshold: overlap threshold used in general inference for panoptic segmentation
            metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                segmentation inference
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            sem_seg_postprocess_before_inference: whether to resize the prediction back
                to original input size before semantic segmentation inference or after.
                For high-resolution dataset like Mapillary, resizing predictions before
                inference will cause OOM error.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
        """
        super().__init__(
            backbone=backbone,
            sem_seg_head=sem_seg_head,
            criterion=criterion,
            num_queries=num_queries,
            panoptic_on=panoptic_on,
            object_mask_threshold=object_mask_threshold,
            overlap_threshold=overlap_threshold,
            metadata=metadata,
            size_divisibility=size_divisibility,
            sem_seg_postprocess_before_inference=sem_seg_postprocess_before_inference,
            pixel_mean=pixel_mean,
            pixel_std=pixel_std,
        )
        self.clip_adapter: ClipAdapter = clip_adapter
        self._region_clip_adapter = region_clip_adapter

        self.clip_ensemble: bool = clip_ensemble
        self.clip_ensemble_weight: float = clip_ensemble_weight

    @classmethod
    def from_config(cls, cfg):
        init_kwargs = MaskFormer.from_config(cfg)
        prompt_learner = build_prompt_learner(cfg.MODEL.CLIP_ADAPTER)
        region_clip_adapter = None
        if cfg.MODEL.CLIP_ADAPTER.SEPERATE_ADAPTER:
            log_first_n(
                logging.WARNING,
                "Using different head for region classification and query classification",
            )
            cls_prompt_learner = build_prompt_learner(
                cfg.MODEL.CLIP_ADAPTER.REGION_CLIP_ADAPTER
            )
            region_clip_adapter = MaskFormerClipAdapter(
                cfg.MODEL.CLIP_ADAPTER.REGION_CLIP_ADAPTER.CLIP_MODEL_NAME,
                cls_prompt_learner,
                mask_fill=cfg.MODEL.CLIP_ADAPTER.MASK_FILL,
                mask_expand_ratio=cfg.MODEL.CLIP_ADAPTER.MASK_EXPAND_RATIO,
                mask_thr=cfg.MODEL.CLIP_ADAPTER.MASK_THR,
                mask_matting=cfg.MODEL.CLIP_ADAPTER.MASK_MATTING,
                region_resized=cfg.MODEL.CLIP_ADAPTER.REGION_RESIZED,
            )

        clip_adapter = MaskFormerClipAdapter(
            cfg.MODEL.CLIP_ADAPTER.CLIP_MODEL_NAME,
            prompt_learner,
            mask_fill=cfg.MODEL.CLIP_ADAPTER.MASK_FILL,
            mask_expand_ratio=cfg.MODEL.CLIP_ADAPTER.MASK_EXPAND_RATIO,
            mask_thr=cfg.MODEL.CLIP_ADAPTER.MASK_THR,
            mask_matting=cfg.MODEL.CLIP_ADAPTER.MASK_MATTING,
            region_resized=cfg.MODEL.CLIP_ADAPTER.REGION_RESIZED,
        )
        init_kwargs["clip_adapter"] = clip_adapter
        init_kwargs["region_clip_adapter"] = region_clip_adapter
        init_kwargs["clip_ensemble"] = cfg.MODEL.CLIP_ADAPTER.CLIP_ENSEMBLE
        init_kwargs[
            "clip_ensemble_weight"
        ] = cfg.MODEL.CLIP_ADAPTER.CLIP_ENSEMBLE_WEIGHT

        return init_kwargs

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """
        
        # # # 图像可视化的时候手动将 dataset_name 设置为'coco_2017_test_stuff_sem_seg' dataset_name = 'coco_2017_test_stuff_sem_seg'
        dataset_name = [x["meta"]["dataset_name"] for x in batched_inputs]
        assert len(set(dataset_name)) == 1
        dataset_name = dataset_name[0] 

        # dataset_name = 'coco_2017_test_stuff_sem_seg'

        
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        # # 2026-01-10 18:30---》想办法利用CLIP图像编码器中的视觉特征：将CLIP的视觉特征与Transformer解码器的特征进行融合，并融入文本信息
        # 构建一个流式Transformer解码器，输入有文本信息，CLIP视觉特征和Backbone所提取的多层次特征！！！！！！ # 这里的视觉特征是固定的，我感觉得需要尽心更新
        # # 2026-03-02把视觉特征融到特征提取器中，先不进行融合了，直接把images_clip_feature视觉特征输入到Transformer解码器中，看看效果
        # # images_layer_features[12(层数), 2(batch-size), 1024(h*w)+1, 768(patch)]
        images_vision = torch.stack(images, dim=0)
        # images_clip_feature, images_layer_features = self.clip_adapter.get_image_features(images_vision)
        images_layer_features = None

        images = ImageList.from_tensors(images, self.size_divisibility)
        
        class_names = self.get_class_name_list(dataset_name)
        text_features, text_layer_features = self.clip_adapter.get_text_features(class_names)  # text_features 每个类别名下的特征嵌入向量 【0~51】多了 text_layer_features变量
        
        # 2026-03-03 11:09 获取CLIP文本编码器提取的文本特征,目的时为了进行文本引导融合
        # 2026-01-13 获取对应图像的类别名，然后将其进行映射
        category_to_id = {name: idx for idx, name in enumerate(class_names)}
        image_class_name = []
        for x in batched_inputs:
            x_txt = "datasets/coco/label_name/{}.txt".format(x['file_name'].split("/")[3].split(".")[0])
            with open(x_txt, "r", encoding="utf-8") as f:
                txt_content = f.read().strip()
                read_categories = json.loads(txt_content)
                category_tensor = build_category_tensor(read_categories, category_to_id)
                image_class_name.append(category_tensor)
        image_txt_feature = []
        image_txt_layer_feature = []
        for i, idx_tensor in enumerate(image_class_name):
            image_txt_feature.append(text_features[idx_tensor, :]) # list[0]表示第一张图像,  如果一张图像中仅含有一个类别那么维度就是[1, 512]，含两个类别则是[2, 512]  下面同理
            image_txt_layer_feature.append(text_layer_features[idx_tensor, :, :]) # list[0], [1, 12, 512] [2, 12, 512]
        


        features = self.backbone(images.tensor)
        outputs = self.sem_seg_head(features, text_layer_features, images_layer_features, images_vision.shape, image_txt_feature) # 'res5' [2, 2048, 16, 16] [2, 1024, 32, 32] [2, 512, 64, 64] [2, 256, 128, 128]
        
        outputs["pred_logits"] = self.clip_adapter.get_sim_logits(
            text_features, self.clip_adapter.normalize_feature(outputs["pred_logits"])
        )
        if self.training:
            if "aux_outputs" in outputs.keys():
                for i in range(len(outputs["aux_outputs"])):
                    outputs["aux_outputs"][i][
                        "pred_logits"
                    ] = self.clip_adapter.get_sim_logits(
                        text_features,
                        self.clip_adapter.normalize_feature(
                            outputs["aux_outputs"][i]["pred_logits"]
                        ),
                    )
            # mask classification target
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                targets = self.prepare_targets(gt_instances, images)
            else:
                targets = None

            # bipartite matching-based loss
            losses = self.criterion(outputs, targets)

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)

            return losses
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

            processed_results = []
            for mask_cls_result, mask_pred_result, input_per_image, image_size in zip(
                mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes
            ):
                height = image_size[0]
                width = image_size[1]
                mask_pred_result = sem_seg_postprocess(
                    mask_pred_result, image_size, height, width
                )
                image = input_per_image["image"].to(self.device)
                # semantic segmentation inference
                r = self.semantic_inference(
                    mask_cls_result, mask_pred_result, image, class_names, dataset_name
                )
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = sem_seg_postprocess(r, image_size, height, width)
                processed_results.append({"sem_seg": r})

                # panoptic segmentation inference
                if self.panoptic_on:
                    panoptic_r = self.panoptic_inference(
                        mask_cls_result, mask_pred_result
                    )
                    processed_results[-1]["panoptic_seg"] = panoptic_r

            return processed_results

    def semantic_inference(self, mask_cls, mask_pred, image, class_names, dataset_name):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        # get the classification result from clip model
        if self.clip_ensemble:
            clip_cls, valid_flag = self.region_clip_adapter(
                image, class_names, mask_pred, normalize=True
            )
            if clip_cls is None:
                clip_cls = torch.empty(0, mask_cls.shape[-1] + 1, device=self.device)
            # softmax before index or after?
            clip_cls = F.softmax(clip_cls[:, :-1], dim=-1)
            if self.clip_ensemble_weight > 0:
                map_back_clip_cls = mask_cls.new_ones(mask_cls.shape)
                
                # 2026-03-03 11:26 注释的下述代码在原始代码中没有
                # # 报了数据不匹配的问题了
                # if clip_cls is not None:
                #     # 获取有效位置的数量
                #     valid_num = valid_flag.sum().item()
                #     clip_cls_num = clip_cls.shape[0]

                #     # 处理形状不匹配的情况
                #     if valid_num != clip_cls_num:
                #         import warnings
                #         warnings.warn(f"Valid class num ({valid_num}) != CLIP class num ({clip_cls_num}), auto aligning...")

                #         # 情况1: CLIP类别数少于有效类别数 → 填充0
                #         if clip_cls_num < valid_num:
                #             pad_num = valid_num - clip_cls_num
                #             pad_tensor = torch.zeros((pad_num, clip_cls.shape[1]), device=clip_cls.device, dtype=clip_cls.dtype)
                #             clip_cls = torch.cat([clip_cls, pad_tensor], dim=0)
                #         # 情况2: CLIP类别数多于有效类别数 → 截断
                #         else:
                #             clip_cls = clip_cls[:valid_num]

                map_back_clip_cls[valid_flag] = clip_cls
                if hasattr(MetadataCatalog.get(dataset_name), "trainable_flag"):
                    trained_mask = torch.Tensor(
                        MetadataCatalog.get(dataset_name).trainable_flag
                    ).to(mask_cls.device)[None, :]
                else:
                    trained_mask = mask_cls.new_zeros(mask_cls.shape)
                mask_cls = trained_mask * torch.pow(
                    mask_cls, self.clip_ensemble_weight
                ) * torch.pow(map_back_clip_cls, 1 - self.clip_ensemble_weight) + (
                    1 - trained_mask
                ) * torch.pow(
                    mask_cls, 1 - self.clip_ensemble_weight
                ) * torch.pow(
                    map_back_clip_cls, self.clip_ensemble_weight
                )
            else:
                # only clip model predictions are used
                mask_cls = clip_cls
                mask_pred = mask_pred[valid_flag]
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg

    def get_class_name_list(self, dataset_name):
        class_names = [
            c.strip() for c in MetadataCatalog.get(dataset_name).stuff_classes
        ]
        return class_names

    @property
    def region_clip_adapter(self):
        if self._region_clip_adapter is None:
            return self.clip_adapter
        return self._region_clip_adapter
