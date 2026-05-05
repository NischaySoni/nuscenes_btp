# ------------------------------------------------------------------
# Focal Loss for Multi-Class Classification
#
# From "Focal Loss for Dense Object Detection" (Lin et al., ICCV 2017)
# Dynamically down-weights well-classified examples so the model
# focuses on hard, misclassified minority classes.
#
# FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
# ------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss with optional label smoothing.

    Args:
        gamma (float): Focusing parameter. Higher gamma = more focus on hard examples.
                        gamma=0 is equivalent to standard Cross-Entropy.
                        Typical values: 1.0 - 3.0 (default: 2.0)
        alpha (float or None): Class balancing weight. If None, all classes weighted equally.
        label_smoothing (float): Label smoothing factor (0.0 = no smoothing).
        reduction (str): 'sum', 'mean', or 'none'.
    """

    def __init__(self, gamma=2.0, alpha=None, label_smoothing=0.0, reduction='sum'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Args:
            logits:  (N, C) raw unnormalized scores
            targets: (N,)   integer class indices
        Returns:
            Focal loss scalar (or per-sample if reduction='none')
        """
        num_classes = logits.size(1)

        # Standard log-softmax for numerical stability
        log_probs = F.log_softmax(logits, dim=1)  # (N, C)
        probs = torch.exp(log_probs)               # (N, C)

        # Gather the probability of the true class
        targets_one_hot = F.one_hot(targets, num_classes).float()  # (N, C)

        # Apply label smoothing if requested
        if self.label_smoothing > 0:
            targets_smooth = targets_one_hot * (1.0 - self.label_smoothing) + \
                             self.label_smoothing / num_classes
        else:
            targets_smooth = targets_one_hot

        # p_t = probability assigned to the true class
        p_t = (probs * targets_one_hot).sum(dim=1)  # (N,)

        # Focal modulating factor: (1 - p_t)^gamma
        focal_weight = (1.0 - p_t) ** self.gamma  # (N,)

        # Cross-entropy part: -sum_c(target_c * log(p_c))
        ce = -(targets_smooth * log_probs).sum(dim=1)  # (N,)

        # Focal loss = focal_weight * CE
        loss = focal_weight * ce  # (N,)

        if self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'mean':
            return loss.mean()
        else:
            return loss
