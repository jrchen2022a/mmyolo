# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Sequence, Union

import torch
from torch import Tensor


def make_divisible(x: float,
                   widen_factor: float = 1.0,
                   divisor: int = 8) -> int:
    """Make sure that x*widen_factor is divisible by divisor."""
    return math.ceil(x * widen_factor / divisor) * divisor


def make_round(x: float, deepen_factor: float = 1.0) -> int:
    """Make sure that x*deepen_factor becomes an integer not less than 1."""
    return max(round(x * deepen_factor), 1) if x > 1 else x


def gt_instances_preprocess(batch_gt_instances: Union[Tensor, Sequence],
                            batch_size: int) -> Tensor:
    """Split batch_gt_instances with batch size, from [all_gt_bboxes, 6] to.

    [batch_size, number_gt, 5]. If some shape of single batch smaller than
    gt bbox len, then using [-1., 0., 0., 0., 0.] to fill.

    Args:
        batch_gt_instances (Sequence[Tensor]): Ground truth
            instances for whole batch, shape [all_gt_bboxes, 6]
        batch_size (int): Batch size.

    Returns:
        Tensor: batch gt instances data, shape [batch_size, number_gt, 5]
    """
    if isinstance(batch_gt_instances, Sequence):
        max_gt_bbox_len = max(
            [len(gt_instances) for gt_instances in batch_gt_instances])
        # fill [0., 0., 0., 0., 0.] if some shape of
        # single batch not equal max_gt_bbox_len
        batch_instance_list = []
        for index, gt_instance in enumerate(batch_gt_instances):
            bboxes = gt_instance.bboxes
            labels = gt_instance.labels
            batch_instance_list.append(
                torch.cat((labels[:, None], bboxes), dim=-1))

            if bboxes.shape[0] >= max_gt_bbox_len:
                continue

            fill_tensor = bboxes.new_full(
                [max_gt_bbox_len - bboxes.shape[0], 5], 0)
            batch_instance_list[index] = torch.cat(
                (batch_instance_list[index], fill_tensor), dim=0)

        return torch.stack(batch_instance_list)
    else:
        # faster version
        # format of batch_gt_instances:
        # [img_ind, cls_ind, x1, y1, x2, y2]

        # sqlit batch gt instance [all_gt_bboxes, 6] ->
        # [batch_size, max_gt_bbox_len, 5]
        assert isinstance(batch_gt_instances, Tensor)
        if len(batch_gt_instances) > 0:
            gt_images_indexes = batch_gt_instances[:, 0]
            max_gt_bbox_len = gt_images_indexes.unique(
                return_counts=True)[1].max()
            # fill [0., 0., 0., 0., 0.] if some shape of
            # single batch not equal max_gt_bbox_len
            batch_instance = torch.zeros((batch_size, max_gt_bbox_len, 5),
                                         dtype=batch_gt_instances.dtype,
                                         device=batch_gt_instances.device)

            for i in range(batch_size):
                match_indexes = gt_images_indexes == i
                gt_num = match_indexes.sum()
                if gt_num:
                    batch_instance[i, :gt_num] = batch_gt_instances[
                        match_indexes, 1:]
        else:
            batch_instance = torch.zeros((batch_size, 0, 5),
                                         dtype=batch_gt_instances.dtype,
                                         device=batch_gt_instances.device)

        return batch_instance

def channel_shuffle(x, groups):
    """Channel Shuffle operation.

    This function enables cross-group information flow for multiple groups
    convolution layers.

    Args:
        x (Tensor): The input tensor.
        groups (int): The number of groups to divide the input tensor
            in the channel dimension.

    Returns:
        Tensor: The output tensor after channel shuffle operation.
    """

    batch_size, num_channels, height, width = x.size()
    assert (num_channels % groups == 0), ('num_channels should be '
                                          'divisible by groups')
    channels_per_group = num_channels // groups

    x = x.view(batch_size, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batch_size, -1, height, width)

    return x
