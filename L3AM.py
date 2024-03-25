# Code for Paper:
# [Title]  - "L3AM: Linear Adaptive Additive Angular Margin Loss for Video-based Hand Gesture Authentication"
# [Author] - Wenwei Song, Wenxiong Kang, Adams Wai-Kin Kong, Yufeng Zhang and Yitao Qiao
# [Github] - https://github.com/SCUT-BIP-Lab/L3AM

import torch
import torch.nn as nn
import numpy as np
import random

def set_random_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
class L3AM(nn.Module):
    # original implementation of Linear Adaptive Additive Angular Margin Loss
    def __init__(self, in_feats, n_classes, s=30):
        super(L3AM, self).__init__()
        self.n_classes = n_classes
        self.in_feats = in_feats
        self.tril_indices = torch.tril_indices(n_classes, n_classes, -1)

        self.W = torch.nn.Parameter(torch.randn(in_feats, n_classes), requires_grad=True)
        nn.init.xavier_normal_(self.W, gain=1)
        # nn.init.xavier_uniform_(self.W)

        self.s = s
        self.eps = 1e-7
        self.pi = np.pi
        self.ce = nn.CrossEntropyLoss()

    def forward(self, x, lb):
        # Normalize input features and proxies according to NormFace
        x_norm = torch.div(x, torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-12))
        w_norm = torch.div(self.W, torch.norm(self.W, p=2, dim=0, keepdim=True).clamp(min=1e-12))

        # Calculate angles between sample features and proxies
        costh = torch.mm(x_norm, w_norm)
        costh = costh.clamp(-1 + self.eps, 1 - self.eps)  # for numerical stability
        theta = torch.acos(costh)

        # Extract the angles (e.g., theta_l in Eq.20) between sample features and their corresponding proxies
        theta_l = theta.gather(dim=1, index=lb.view(-1,1)).view(-1)

        # Calculate the cosine similarity among proxies
        cosww = torch.mm(w_norm.transpose(0, 1), w_norm).clamp(min=-1.0, max=1.0)
        cosww = cosww - torch.diag(torch.tensor(float("inf")).repeat(self.n_classes), 0)

        # Find the minimum interclass angle (e.g., Eq.17)
        cosww_max = cosww.max(dim=1)[0]
        cosww_max_l = cosww_max.gather(dim=0, index=lb)
        beta_min = torch.acos(cosww_max_l)

        with torch.no_grad():
            # Eq.17
            m_c = beta_min - 2 * theta_l
            # Eq.19
            m_delta = (0.6 - m_c) / 2.
            m_delta = m_delta.clamp(min=0., max=0.1)
            m = torch.clamp(m_c + m_delta, min=0.4)

        one_hot = torch.zeros_like(costh)
        one_hot.scatter_(1, lb.view(-1, 1), 1)

        # Eq.20
        target = self.pi / 2. - (theta + m.view(-1,1))
        others = self.pi / 2. - theta

        output = (one_hot * target) + ((1.0 - one_hot) * others)
        output *= self.s  # see NormFace
        loss = self.ce(output, lb)

        return loss


class L3AM_Efficient(nn.Module):
    # implementation of Linear Adaptive Additive Angular Margin Loss
    def __init__(self, in_feats, n_classes, s=30):
        super(L3AM_Efficient, self).__init__()
        self.n_classes = n_classes
        self.in_feats = in_feats

        self.W = torch.nn.Parameter(torch.randn(in_feats, n_classes), requires_grad=True)
        nn.init.xavier_normal_(self.W, gain=1)
        # nn.init.xavier_uniform_(self.W)

        self.s = s
        self.eps = 1e-7
        self.pi = np.pi
        self.ce = nn.CrossEntropyLoss()

    def forward(self, x, lb):
        # Normalize input features and proxies according to NormFace
        x_norm = torch.div(x, torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-12))
        w_norm = torch.div(self.W, torch.norm(self.W, p=2, dim=0, keepdim=True).clamp(min=1e-12))

        # Calculate angles between sample features and proxies
        costh = torch.mm(x_norm, w_norm)
        costh = costh.clamp(-1 + self.eps, 1 - self.eps)  # for numerical stability
        theta = torch.acos(costh)

        # Extract the angles (e.g., theta_l in Eq.20) between sample features and their corresponding proxies
        theta_l = theta.gather(dim=1, index=lb.view(-1, 1)).view(-1)
        # Extract the samples' corresponding proxies
        w_norm_l = w_norm[:, lb].permute(1,0)

        # Calculate the cosine similarity between the corresponding proxies and other proxies
        cosww = torch.mm(w_norm_l, w_norm).clamp(min=-1.0, max=1.0)
        cosww.scatter_(dim=1, index=lb.view(-1, 1), value=-float("inf"))

        # Find the minimum interclass angle (e.g., Eq.17)
        cosww_o_max = cosww.max(dim=1)[0]
        beta_min = torch.acos(cosww_o_max)

        with torch.no_grad():
            # Eq.17
            m_a = beta_min - 2 * theta_l
            # Eq.19
            m_delta = (0.6 - m_a) / 2.
            m_delta = m_delta.clamp(min=0., max=0.1)
            m = torch.clamp(m_a + m_delta, min=0.4)

        # Eq.20
        output = self.pi / 2. - theta
        theta_l_m = self.pi / 2. - (theta_l + m)
        output.scatter_(dim=1, index=lb.view(-1, 1), src=theta_l_m.view(-1, 1))
        output *= self.s # see NormFace
        loss = self.ce(output, lb)

        return loss


if __name__ == '__main__':

    # The dimension of the feature input to the loss function is 512
    in_feats = 512
    # The number of classification categories is 143
    n_classes = 143
    batch_size = 6

    input = torch.rand((batch_size, in_feats))
    label = torch.randint(0, n_classes, size=(batch_size,))

    set_random_seed()
    L3AM = L3AM(in_feats, n_classes)
    loss1 = L3AM(input, label)

    set_random_seed()
    L3AM_Efficient = L3AM_Efficient(in_feats, n_classes)
    loss2 = L3AM_Efficient(input, label)

    if loss1 == loss2:
        print("L3AM and L3AM_Efficient have the same loss values. \nL3AM is the loss function used in the experiments of the IJCV paper. \nL3AM_Efficient optimizes L3AM for efficiency, but is not fully tested.")

