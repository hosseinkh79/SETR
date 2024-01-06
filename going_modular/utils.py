import torch


def intersection_over_union_multiclass(predicted, target):
    # Assuming predicted and target have shapes (batch_size, num_classes, width, height)

    intersection = torch.sum(predicted * target)
    union = torch.sum(predicted) + torch.sum(target) - intersection
    
    iou = (intersection + 1e-15) / (union + 1e-15)  # Adding a small epsilon to avoid division by zero
    
    return iou
