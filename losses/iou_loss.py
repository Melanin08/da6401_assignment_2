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
        # Use ReLU instead of clamp (better gradients)
        pred_w = torch.relu(pred_boxes[:, 2])
        pred_h = torch.relu(pred_boxes[:, 3])
        target_w = torch.relu(target_boxes[:, 2])
        target_h = torch.relu(target_boxes[:, 3])

        # Convert to corner format
        pred_x1 = pred_boxes[:, 0] - pred_w / 2
        pred_y1 = pred_boxes[:, 1] - pred_h / 2
        pred_x2 = pred_boxes[:, 0] + pred_w / 2
        pred_y2 = pred_boxes[:, 1] + pred_h / 2

        target_x1 = target_boxes[:, 0] - target_w / 2
        target_y1 = target_boxes[:, 1] - target_h / 2
        target_x2 = target_boxes[:, 0] + target_w / 2
        target_y2 = target_boxes[:, 1] + target_h / 2

        # Intersection
        inter_x1 = torch.max(pred_x1, target_x1)
        inter_y1 = torch.max(pred_y1, target_y1)
        inter_x2 = torch.min(pred_x2, target_x2)
        inter_y2 = torch.min(pred_y2, target_y2)

        inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
        inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
        inter_area = inter_w * inter_h

        # Areas (robust)
        pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
        target_area = (target_x2 - target_x1) * (target_y2 - target_y1)

        # Union
        union_area = pred_area + target_area - inter_area

        # IoU
        iou = inter_area / (union_area + self.eps)

        loss = 1 - iou

        if self.reduction == "mean":
            return loss.mean()
        return loss.sum()
