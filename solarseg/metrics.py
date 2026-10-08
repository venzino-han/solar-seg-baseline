import torch


@torch.no_grad()
def evaluate(model, loader, device, amp_dtype=None, thr=0.5):
    """Pixel metrics over the whole split.

    iou       = sum(intersection) / sum(union) over all frames   (dataset-level IoU)
    dice      = mean of per-frame Dice (a frame with empty GT and empty prediction scores 1)
    precision / recall / f1 = pixel-level, accumulated over all frames
    """
    model.eval()
    tp = fp = fn = inter = union = dice_sum = 0.0
    n = empty_gt = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            p = (torch.sigmoid(model(x).float()) > thr).float()
        tp += (p * y).sum().item(); fp += (p * (1 - y)).sum().item(); fn += ((1 - p) * y).sum().item()
        i = (p * y).sum((1, 2, 3)); u = ((p + y) > 0).float().sum((1, 2, 3))
        inter += i.sum().item(); union += u.sum().item()
        dice_sum += ((2 * i + 1) / (p.sum((1, 2, 3)) + y.sum((1, 2, 3)) + 1)).sum().item()
        empty_gt += int((y.sum((1, 2, 3)) == 0).sum().item()); n += x.size(0)
    prec = tp / (tp + fp + 1e-6); rec = tp / (tp + fn + 1e-6)
    return {"iou": inter / (union + 1e-6), "dice": dice_sum / max(1, n), "precision": prec, "recall": rec,
            "f1": 2 * prec * rec / (prec + rec + 1e-6), "frames": n, "frames_empty_gt": empty_gt}
