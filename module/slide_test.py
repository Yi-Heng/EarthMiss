import torch
import ever as er
import torch.nn.functional as F
from torch import Tensor
from typing import List

def slide_inference(model, inputs: Tensor, y:dict,
                    slide_config: List[dict]) -> Tensor:
    """Inference by sliding-window with overlap.

    If h_crop > h_img or w_crop > w_img, the small patch will be used to
    decode without padding.

    Args:
        inputs (tensor): the tensor should have a shape NxCxHxW,
            which contains all images in the batch.
        batch_img_metas (List[dict]): List of image metainfo where each may
            also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
            'ori_shape', and 'pad_shape'.
            For details on the values of these keys see
            `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

    Returns:
        Tensor: The segmentation results, seg_logits from model of each
            input image.
    """

    h_stride, w_stride = slide_config['stride']
    h_crop, w_crop = slide_config['crop_size']
    batch_size, _, h_img, w_img = inputs.size()
    try:
        out_channels = model.module.config.num_classes if model.module.config.num_classes != 1 else 2
    except:
        out_channels = model.config.num_classes if model.config.num_classes != 1 else 2
    h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
    w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
    preds = inputs.new_zeros((batch_size, out_channels, h_img, w_img))
    count_mat = inputs.new_zeros((batch_size, 1, h_img, w_img))
    for h_idx in range(h_grids):
        for w_idx in range(w_grids):
            y1 = h_idx * h_stride
            x1 = w_idx * w_stride
            y2 = min(y1 + h_crop, h_img)
            x2 = min(x1 + w_crop, w_img)
            y1 = max(y2 - h_crop, 0)
            x1 = max(x2 - w_crop, 0)
            crop_img = inputs[:, :, y1:y2, x1:x2]
            # change the image shape to patch shape
            # the output of encode_decode is seg logits tensor map
            # with shape [N, C, H, W]
            crop_seg_logit = model(crop_img,y)
            preds += F.pad(crop_seg_logit,
                            (int(x1), int(preds.shape[3] - x2), int(y1),
                            int(preds.shape[2] - y2)))

            count_mat[:, :, y1:y2, x1:x2] += 1
    assert (count_mat == 0).sum() == 0
    seg_logits = preds / count_mat

    return seg_logits