import torch
import torch.nn as nn
import torch.nn.functional as F


def geo_scal_loss(pred, ssc_target, ignore_index=255, non_empty_idx=0):
    pred = F.softmax(pred, dim=1)

    empty_probs = pred[:, non_empty_idx]
    nonempty_probs = 1.0 - empty_probs

    mask = ssc_target != ignore_index
    nonempty_target = (ssc_target != non_empty_idx) & mask
    nonempty_target = nonempty_target.float()
    nonempty_probs = nonempty_probs[mask]
    empty_probs = empty_probs[mask]
    nonempty_target = nonempty_target[mask]

    eps = 1e-5
    intersection = (nonempty_target * nonempty_probs).sum()
    precision = intersection / (nonempty_probs.sum() + eps)
    recall = intersection / (nonempty_target.sum() + eps)
    spec = ((1.0 - nonempty_target) * empty_probs).sum() / ((1.0 - nonempty_target).sum() + eps)

    return (
        F.binary_cross_entropy(precision, torch.ones_like(precision))
        + F.binary_cross_entropy(recall, torch.ones_like(recall))
        + F.binary_cross_entropy(spec, torch.ones_like(spec))
    )


def sem_scal_loss(pred, ssc_target, ignore_index=255):
    pred = F.softmax(pred, dim=1)
    loss = 0.0
    count = 0.0
    mask = ssc_target != ignore_index
    n_classes = pred.shape[1]

    for i in range(n_classes):
        p = pred[:, i]
        p = p[mask]
        target = ssc_target[mask]

        completion_target = torch.ones_like(target)
        completion_target[target != i] = 0
        if torch.sum(completion_target) > 0:
            count += 1.0
            nominator = torch.sum(p * completion_target)
            loss_class = 0.0
            if torch.sum(p) > 0:
                precision = nominator / torch.sum(p)
                loss_class += F.binary_cross_entropy(precision, torch.ones_like(precision))
            if torch.sum(completion_target) > 0:
                recall = nominator / torch.sum(completion_target)
                loss_class += F.binary_cross_entropy(recall, torch.ones_like(recall))
            if torch.sum(1 - completion_target) > 0:
                specificity = torch.sum((1 - p) * (1 - completion_target)) / torch.sum(1 - completion_target)
                loss_class += F.binary_cross_entropy(specificity, torch.ones_like(specificity))
            loss += loss_class

    if count == 0:
        return pred.sum() * 0.0
    return loss / count


def CE_ssc_loss(pred, target, class_weights=None, ignore_index=255):
    criterion = nn.CrossEntropyLoss(
        weight=class_weights, ignore_index=ignore_index, reduction="mean"
    )
    return criterion(pred, target.long())


def scan_loss(pred, target, ignore_index=255):
    # pred: [B, C, X, Y, Z], target: [B, X, Y, Z]
    b, c, xh, yw, zd = pred.shape
    device = pred.device

    # cumulative averages from prediction logits
    pred_cumsum_height = torch.cumsum(pred, dim=4)
    pred_cumsum_height = pred_cumsum_height / torch.arange(1, zd + 1, device=device).view(1, 1, 1, 1, zd)

    pred_cumsum_depth = torch.cumsum(pred.flip((2,)), dim=2)
    pred_cumsum_depth = pred_cumsum_depth / torch.arange(1, xh + 1, device=device).view(1, 1, xh, 1, 1)

    # cumulative averages from target
    target_occ = target.clone()
    target_occ[target_occ == ignore_index] = c
    one_hot = F.one_hot(target_occ.long(), num_classes=c + 1).permute(0, 4, 1, 2, 3).float()
    one_hot = one_hot[:, :c]

    target_cum_height = torch.cumsum(one_hot, dim=4)
    target_cum_height = target_cum_height / torch.arange(1, zd + 1, device=device).view(1, 1, 1, 1, zd)

    target_cum_depth = torch.cumsum(one_hot.flip((2,)), dim=2)
    target_cum_depth = target_cum_depth / torch.arange(1, xh + 1, device=device).view(1, 1, xh, 1, 1)

    loss_height = F.cross_entropy(pred_cumsum_height.float(), target_cum_height, reduction="mean")
    loss_depth = F.cross_entropy(pred_cumsum_depth.float(), target_cum_depth, reduction="mean")
    return loss_height + loss_depth
