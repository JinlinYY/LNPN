import torch
import torch.nn as nn
import torch.nn.functional as F

# 示例使用Focal Loss的初始化
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)                          # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        # logpt = F.log_softmax(input)
        logpt = F.log_softmax(input, dim=1)

        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = logpt.data.exp()

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * at

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

# 示例使用Focal Loss的初始化
# loss_func = FocalLoss(gamma=2)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# class EnhancedFocalLoss(nn.Module):
#     def __init__(self, gamma=2, alpha=None, epsilon=1e-7, weight=None):
#         """
#         参数:
#         gamma: Focal Loss的焦点参数。默认为2。
#         alpha: 类别权重，可以是单个值（适用于二分类）或包含每个类别权重的列表（适用于多分类）。
#         epsilon: 防止log运算时的数值不稳定问题。
#         weight: 类别平衡的权重。
#         """
#         super(EnhancedFocalLoss, self).__init__()
#         self.gamma = gamma
#         self.alpha = alpha
#         self.epsilon = epsilon
#         self.weight = weight
#
#         if isinstance(alpha, (float, int)):
#             self.alpha = torch.tensor([alpha, 1 - alpha])
#         if isinstance(alpha, list):
#             self.alpha = torch.tensor(alpha)
#
#     def forward(self, input, target):
#         if input.dim() > 2:
#             input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
#             input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
#             input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
#
#         target = target.view(-1, 1)
#
#         logpt = F.log_softmax(input, dim=1)
#         logpt = logpt.gather(1, target)
#         logpt = logpt.view(-1)
#         pt = logpt.exp()
#
#         # Apply alpha weights
#         if self.alpha is not None:
#             if self.alpha.type() != input.data.type():
#                 self.alpha = self.alpha.type_as(input.data)
#             at = self.alpha.gather(0, target.data.view(-1))
#             logpt = logpt * at
#
#         # Focal loss component
#         focal_loss = -1 * (1 - pt) ** self.gamma * logpt
#
#         # Weighted loss component
#         if self.weight is not None:
#             focal_loss *= self.weight[target.squeeze().long()]
#
#         return focal_loss.mean()


# # 使用示例
# # Assuming classes 0, 1, 2 have weights 0.1, 0.5, 0.4
# alpha = [0.1, 0.5, 0.4]
# # Example class weights computed as inverses of class frequencies
# class_weights = torch.tensor([0.5, 2.0, 1.0]).to(device)
# loss_func = EnhancedFocalLoss(gamma=2, alpha=alpha, weight=class_weights)


class AdaptiveWeightedFocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None):
        super(AdaptiveWeightedFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)
            input = input.transpose(1, 2)
            input = input.contiguous().view(-1, input.size(2))
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = logpt.exp()

        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_tensor = torch.tensor([self.alpha, 1 - self.alpha]).to(input.device)
            else:
                alpha_tensor = torch.tensor(self.alpha).to(input.device)
            alpha = alpha_tensor.gather(0, target.view(-1))
            logpt = logpt * alpha

        # 自适应权重
        loss = -1 * (1 - pt) ** (self.gamma / (1 + pt)) * logpt
        return loss.mean()

