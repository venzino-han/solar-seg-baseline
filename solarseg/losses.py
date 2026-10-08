import torch
import torch.nn.functional as F


def dice_loss(logits, target, eps=1.0):
    p = torch.sigmoid(logits)
    num = 2 * (p * target).sum((1, 2, 3)) + eps
    den = p.sum((1, 2, 3)) + target.sum((1, 2, 3)) + eps
    return (1 - num / den).mean()


def seg_loss(logits, target, pos_weight):
    """BCE with a positive-class weight (handles the tiny foreground fraction) + soft Dice."""
    return F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight) + dice_loss(logits, target)
