import torch 
import torch.nn as nn 

def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy 

class CrossEntropyLabelSmooth(nn.Module):
    
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """      

    def __init__(self, num_classes, epsilon=0.1, reduction=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.reduction = reduction
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets, applied_softmax=False):
        """
        Args:
            inputs: prediction matrix (after softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (batch_size, num_classes).
        """
        if applied_softmax:
            log_probs = torch.log(inputs)
        else:
            log_probs = self.logsoftmax(inputs)
        
        if inputs.shape != targets.shape:
            # this means that the target data shape is (B,)
            targets = torch.zeros_like(inputs).scatter(1, targets.unsqueeze(1), 1)
        
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).sum(dim=1)
        # print(loss)
        if self.reduction:
            # print(loss)
            return loss.mean()
        else:
            return loss