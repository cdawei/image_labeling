import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F


def criterion_rpc(true_labels, pred_scores, p=1, debug=False):
    """
    Generalised reversed P-Classification with Squared Hinge Loss:
    \ell_+(v) = 1/p * [1 - pv]_+^2 = [1/sqrt(p) - sqrt(p) * v]_+^2
    \ell_-(v) = [1 + v]_+^2
    """
    if debug:
        assert len(true_labels.shape) == 2
        assert true_labels.shape == pred_scores.shape
        assert p > 0
    Yp = (true_labels).float()
    Yn = (1 - true_labels).float()
    sqrt_p = np.sqrt(p)
    
    T1p = F.relu(1 / sqrt_p - sqrt_p * pred_scores) * Yp
    T1n = F.relu(1 + pred_scores) * Yn
    
    T2p = T1p * T1p
    T2n = T1n * T1n
    
    T = (T2p + T2n).sum(dim=1)
    return T.mean()



def criterion_gpc_sh(true_labels, pred_scores, p=3, debug=False):
    """
    Another Generalised P-Classification with Squared Hinge Loss:
    \ell_+(v) = [1 - v]_+^2 
    \ell_-(v) = p * [1 + v/p]_+^2 = [sqrt(p) + v/sqrt(p)]_+^2
    """
    assert len(true_labels.shape) == 2
    assert true_labels.shape == pred_scores.shape
    Yp = (true_labels).float()
    Yn = (1 - true_labels).float()
    
    if debug:
        print('%.4f, %.4f' % (pred_scores.min().item(), pred_scores.max().item()))
    
    T1p = F.relu(1 - pred_scores) * Yp
    T1n = F.relu(p * (1 + pred_scores)) * Yn
    
    T2p = T1p * T1p
    T2n = T1n * T1n
    
    if debug:
        print('%.4f, %.4f' % (T2p.min().item(), T2p.max().item()))
        print('%.4f, %.4f' % (T2n.min().item(), T2n.max().item()))
    
    T = T2p.sum(dim=1) + T2n.sum(dim=1) / p
    return T.mean()


def criterion_pcg(true_labels, pred_scores, p=3, debug=False):
    """
    Generalised P-Classification with Squared Hinge Loss:
    \ell_+(v) = [1 - v]_+^2 
    \ell_-(v) = 1/p * [1 + pv]_+^2 = [1/sqrt(p) + sqrt(p) * v]_+^2
    """
    assert len(true_labels.shape) == 2
    assert true_labels.shape == pred_scores.shape
    assert p > 0
    Yp = (true_labels).float()
    Yn = (1 - true_labels).float()
    sqrt_p = np.sqrt(p)
    
    if debug:
        print('%.4f, %.4f' % (pred_scores.min().item(), pred_scores.max().item()))
    
    T1p = F.relu(1 - pred_scores) * Yp
    # T1n = F.relu(1 + p * pred_scores) * Yn
    T1n = F.relu(1 / sqrt_p + sqrt_p * pred_scores) * Yn
    
    T2p = T1p * T1p
    T2n = T1n * T1n
    
    if debug:
        print('%.4f, %.4f' % (T2p.min().item(), T2p.max().item()))
        print('%.4f, %.4f' % (T2n.min().item(), T2n.max().item()))
    
    # T = T2p.sum(dim=1) + T2n.sum(dim=1) / p
    T = (T2p + T2n).sum(dim=1)
    return T.mean()


def criterion_pc(true_labels, pred_scores, p=3, debug=False):
    """
    P-Classification loss: 
    """
    assert len(true_labels.shape) == 2
    assert true_labels.shape == pred_scores.shape
    Yp = (true_labels).float()
    Yn = (1 - true_labels).float()
    
    if debug:
        print('%.4f, %.4f' % (pred_scores.min().item(), pred_scores.max().item()))
    
    T1p = pred_scores * Yp
    T1n = pred_scores * Yn
    
    T2p = torch.exp(-T1p)
    T2n = torch.exp(p * T1n)
    
    if debug:
        print('%.4f, %.4f' % (T2p.min().item(), T2p.max().item()))
        print('%.4f, %.4f' % (T2n.min().item(), T2n.max().item()))
    
    Tp = T2p * Yp
    Tn = T2n * Yn
    
    T = Tp.sum(dim=1) + Tn.sum(dim=1) / p
    return T.mean()


class mymodel(nn.Module):
    def __init__(self, num_labels, num_factors):
        super(mymodel, self).__init__()
        self.num_labels = num_labels
        self.num_factors = num_factors
        self.resnet = torchvision.models.resnet152(pretrained=True)
        self.gpu_device = None

        # freeze all the network except the final layer
        for param in self.resnet.parameters():
            param.requires_grad = False

        # Parameters of newly constructed modules have requires_grad=True by default
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, self.num_factors)

        # label factors
        # self.label_embeds = nn.Embedding(self.num_labels, self.num_factors)
        # self.lookup_tensor = torch.arange(self.num_labels)
        self.label_factors = nn.Parameter(torch.zeros(self.num_labels, self.num_factors))
        self.label_biases = nn.Parameter(torch.zeros(self.num_labels, 1))
        nn.init.xavier_uniform_(self.label_factors)
        nn.init.xavier_uniform_(self.label_biases)

    def forward(self, x):
        rs_outputs = self.resnet(x)  # batch_size x num_factors
        # label_factors = self.label_embeds(self.lookup_tensor)  # num_labels x num_factors
        # scores = torch.matmul(F.softmax(rs_outputs, dim=1), label_factors.t())  # batch_size x num_labels
        # scores = torch.matmul(rs_outputs, label_factors.t()) / 100  # scale scores to prevent numerical overflow in criterion_pc()
        
        scores = torch.matmul(rs_outputs, self.label_factors.t()) + self.label_biases.t()
        return scores
    
    # def parameters(self):
        # return list(self.resnet.fc.parameters()) + list(self.label_embeds.parameters())

    # The to() method transfers all floating point (not integers) data to the target device
    # See https://pytorch.org/docs/stable/_modules/torch/nn/modules/module.html
    def to(self, *args, **kwargs):
        device, _, _ = torch._C._nn._parse_to(*args, **kwargs)
        # self.lookup_tensor = self.lookup_tensor.to(device)
        self.gpu_device = device
        return super(mymodel, self).to(*args, **kwargs)

    def train(self):
        self.resnet.train()

    def eval(self):
        self.resnet.eval()


# NOTE 
# This is equivalent to `mymodel` due to two consecutive linear layers without non-linear transformation in between (in `mymodel`)
# is equivalent to one linear transformation (in `mymodel1`)
# `mymodel` can be explained from a matrix factorisation perspective
class mymodel1(nn.Module):
    def __init__(self, num_labels):
        super(mymodel, self).__init__()
        self.num_labels = num_labels
        self.resnet = torchvision.models.resnet152(pretrained=True)
        self.gpu_device = None

        # freeze all the network except the final layer
        for param in self.resnet.parameters():
            param.requires_grad = False

        # Parameters of newly constructed modules have requires_grad=True by default
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, self.num_labels)

    def forward(self, x):
        return self.resnet(x)  # batch_size x num_labels

    # The to() method transfers all floating point (not integers) data to the target device
    # See https://pytorch.org/docs/stable/_modules/torch/nn/modules/module.html
    def to(self, *args, **kwargs):
        device, _, _ = torch._C._nn._parse_to(*args, **kwargs)
        self.gpu_device = device
        return super(mymodel, self).to(*args, **kwargs)

    def train(self):
        self.resnet.train()

    def eval(self):
        self.resnet.eval()

