import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from ops import *

def conv(in_planes, out_planes, kernel_size, stride, p_init, fp, spm):
    padding = (kernel_size-1)//2
    if fp: return nn.Conv2d(in_planes, out_planes, kernel_size, stride=stride, padding=padding, bias=False)
    return NoisyConv2d(in_planes, out_planes, kernel_size, stride=stride, padding=padding, bias=False, p_init=p_init, spm=spm)

def linear(in_feats, out_feats, p_init, fp, spm):
    if fp: return nn.Linear(in_feats, out_feats)
    return NoisyLinear(in_feats, out_feats, p_init=p_init, spm='None')

class Shortcut(nn.Module):
  def __init__(self, in_planes, planes, stride):
    super(Shortcut, self).__init__()
    self.avg = nn.AvgPool2d(stride) if stride != 1 else nn.Identity()
    self.in_planes = in_planes
    self.planes = planes

  def forward(self, x):
    x = self.avg(x)

    if self.in_planes != self.planes:
        x = F.pad(x, (0,0,0,0,0,self.planes-self.in_planes))

    return x

class ClipFunc(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x, alpha)
        return torch.clamp(x, min = 0, max = alpha.item())

    @staticmethod
    def backward(ctx, dLdy_q):
        x, alpha, = ctx.saved_tensors
        lower_bound = x < 0
        upper_bound = x > alpha
        x_range = ~(lower_bound|upper_bound)
        grad_alpha = torch.sum(dLdy_q * torch.ge(x, alpha).float()).view(-1)
        return dLdy_q * x_range.float(), grad_alpha

class NoisyConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride, bias, p_init, spm='None'):
        super(NoisyConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.mode = 'noisy'
        self.qf = torch.floor

        self.set_prec = None
        self.spm = spm
        # print("self.spm", self.spm)

        s = math.log(2**(1-p_init) / (1 - 2**(1-p_init)))
        # self.weight_s = nn.Parameter(torch.ones_like(self.weight)*s)
        self.weight_s = nn.Parameter(torch.ones(1, self.weight.shape[1], 1, 1) *s)
        if self.bias is not None:
            self.bias_s = nn.Parameter(torch.ones_like(self.bias)*s)

        # self.p, self.bp = None, None

        self.p = None
        self.bp = None

        self.firstLayer = True
        self.alpha = nn.Parameter(torch.tensor(4.0))

    def _setP(self):
        self.p = createPTensor(self.weight.data, self.weight_s.data, self.set_prec, self.spm)

    def forward(self, x):
        bias = None
        # print("selfpdebug", self.p == None)

        if not self.firstLayer:

            x = torch.clip(x, min=0, max=6)

            clipped = ClipFunc.apply(x, self.alpha)

            if self.mode == 'quant':
                if self.p == None:
                    self._setP()
                # self.xp = self.qf(1-torch.log2(torch.sigmoid(self.weight_s))) # weight_s
                x = Quant.apply(clipped/self.alpha.detach(), self.p) * self.alpha.detach()
            elif self.mode == 'noisy':
                x = clipped + self.alpha * torch.sigmoid(self.weight_s)/2 * torch.empty_like(x).uniform_(-.1, .1)

        if self.mode == 'quant':
            if self.p == None:
                self._setP()
            weight = Quantize.apply(self.weight, self.p, self.qf)
            if self.bias is not None:
                if self.bp == None:
                    self.bp = createPTensor(self.bias.data, self.bias_s, self.set_prec, self.spm)
                bias = Quantize.apply(self.bias, self.bp, self.qf)
        elif self.mode == 'noisy':
            weight = self.weight + torch.sigmoid(self.weight_s) * torch.empty_like(self.weight).uniform_(-1, 1)
            if self.bias is not None:
                bias = self.bias + torch.sigmoid(self.bias_s) * torch.empty_like(self.bias).uniform_(-1, 1)
        out = F.conv2d(x, weight, bias=bias, stride=self.stride, padding=self.padding)

        
        return out

class NoisyLinear(nn.Linear):
    def __init__(self, in_channels, out_channels, p_init, spm='None'):
        super(NoisyLinear, self).__init__(in_channels, out_channels)
        self.mode = 'noisy'
        self.qf = torch.floor

        self.set_prec = None
        self.spm = spm

        self.p = None
        self.bp = None

        s = math.log(2**(1-p_init) / (1 - 2**(1-p_init)))
        self.weight_s = nn.Parameter(torch.ones_like(self.weight)*s)
        if self.bias is not None:
            self.bias_s = nn.Parameter(torch.ones_like(self.bias)*s)

        # self.p, self.bp = None, None

    def forward(self, x):
        bias = None
        if self.mode == 'quant':
            if self.p == None:
                self.p = createPTensor(self.weight.data, self.weight_s.data, self.set_prec, self.spm)
            weight = Quantize.apply(self.weight, self.p, self.qf)
            if self.bias is not None:
                if self.bp == None:
                    self.bp = createPTensor(self.bias.data, self.bias_s, self.set_prec, self.spm)
                bias = Quantize.apply(self.bias, self.bp, self.qf)
        elif self.mode == 'noisy':
            weight = self.weight + torch.sigmoid(self.weight_s) * torch.empty_like(self.weight).uniform_(-1, 1)
            if self.bias is not None:
                bias = self.bias + torch.sigmoid(self.bias_s) * torch.empty_like(self.bias).uniform_(-1, 1)
        out = F.linear(x, weight, bias=bias)
        return out
