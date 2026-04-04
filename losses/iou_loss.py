import torch
import torch.nn as nn


class IoULoss(nn.Module):
    def __init__(self, eps: float = 1e-6, reduction: str = "mean"):
        super().__init__()

        if reduction not in ["mean", "sum"]:
            raise ValueError("Reduction must be 'mean' or 'sum'")

        self.eps = eps
        self.reduction = reduction

    def forward(self, pred_boxes, target_boxes):
        # Keep predicted box sizes non-negative
        pred_w = torch.clamp(pred_boxes[:, 2], min=0.0)
        pred_h = torch.clamp(pred_boxes[:, 3], min=0.0)

        # Keep target box sizes non-negative for safety
        target_w = torch.clamp(target_boxes[:, 2], min=0.0)
        target_h = torch.clamp(target_boxes[:, 3], min=0.0)

        # Convert [x_center, y_center, width, height] -> [x1, y1, x2, y2]
        pred_x1 = pred_boxes[:, 0] - pred_w / 2.0
        pred_y1 = pred_boxes[:, 1] - pred_h / 2.0
        pred_x2 = pred_boxes[:, 0] + pred_w / 2.0
        pred_y2 = pred_boxes[:, 1] + pred_h / 2.0

        target_x1 = target_boxes[:, 0] - target_w / 2.0
        target_y1 = target_boxes[:, 1] - target_h / 2.0
        target_x2 = target_boxes[:, 0] + target_w / 2.0
        target_y2 = target_boxes[:, 1] + target_h / 2.0

        # Intersection
        inter_x1 = torch.max(pred_x1, target_x1)
        inter_y1 = torch.max(pred_y1, target_y1)
        inter_x2 = torch.min(pred_x2, target_x2)
        inter_y2 = torch.min(pred_y2, target_y2)

        inter_w = torch.clamp(inter_x2 - inter_x1, min=0.0)
        inter_h = torch.clamp(inter_y2 - inter_y1, min=0.0)
        inter_area = inter_w * inter_h

        # Areas
        pred_area = torch.clamp(pred_x2 - pred_x1, min=0.0) * torch.clamp(pred_y2 - pred_y1, min=0.0)
        target_area = torch.clamp(target_x2 - target_x1, min=0.0) * torch.clamp(target_y2 - target_y1, min=0.0)

        # Union
        union_area = pred_area + target_area - inter_area + self.eps

        # IoU in [0, 1]
        iou = inter_area / union_area

        # Loss in [0, 1]
        loss = 1.0 - iou

        if self.reduction == "mean":
            return loss.mean()
        return loss.sum()
