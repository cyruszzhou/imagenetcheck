import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from layers import *

def mult_by_shape(val, shape, index=1):
    for i in range(len(shape)):
        if i == index:
            val *= 1
        else:
            val *= shape[i]

    return val 

class MixedNet(nn.Module):
    def __init__(self, mode, qf, p_init):
        super(MixedNet, self).__init__()

        self.mixed_layers = []
        self.mode = mode
        self.qf = qf
        self.p_init = p_init

    def set_mode(self, mode):
        self.mode = mode
        for m in self.mixed_layers: m.mode = mode

    def set_qf(self, qf):
        self.qf = qf
        for m in self.mixed_layers: m.qf = qf

    def prec_cost(self):
        cost = 0
        for m in self.mixed_layers:
            # cost += -torch.log2(torch.sigmoid(m.weight_s)).sum()
            cost += mult_by_shape(-torch.log2(torch.sigmoid(m.weight_s)).sum(), m.weight.shape)
            if m.bias is not None: cost += -torch.log2(torch.sigmoid(m.bias_s)).sum()
        return cost

    def project(self):
        for m in self.mixed_layers:
            # m.weight_s.data = torch.clamp(m.weight_s.data, min=-1.95)

            p = 1-torch.log2(torch.sigmoid(m.weight_s))
            M = 2-2**(1-p)
            m.weight.data = torch.min(m.weight.data, M)
            m.weight.data = torch.max(m.weight.data, -M)
            if m.bias is not None:
                p = 1-torch.log2(torch.sigmoid(m.bias_s))
                M = 2-2**(1-p)
                m.bias.data = torch.min(m.bias.data, M)
                m.bias.data = torch.max(m.bias.data, -M)

    def print_precs(self):
        if self.mode == 'quant':
            return self.print_precs_quantize()

        precs = 0.
        norm_precs = 0.
        num_precs = 0.
        for m in self.mixed_layers:
            scaling = max(1e-6, min(1, torch.abs(m.weight).max().item()))
            noise_mag = torch.sigmoid(m.weight_s)

            p_float = 1-torch.log2(noise_mag)
            p_float = torch.clamp(p_float, min=1)

            norm_p_float = 1-torch.log2(noise_mag/scaling)
            norm_p_float = torch.clamp(norm_p_float, min=1)

            precs += self.qf(p_float).sum()
            norm_precs += self.qf(norm_p_float).sum()
            num_precs += m.weight_s.numel()

            if m.bias is not None:
                scaling = max(1e-6, min(1, torch.abs(m.bias).max().item()))
                noise_mag = torch.sigmoid(m.bias_s)

                p_float = 1-torch.log2(noise_mag)
                p_float = torch.clamp(p_float, min=1)

                norm_p_float = 1-torch.log2(noise_mag/scaling)
                norm_p_float = torch.clamp(norm_p_float, min=1)

                precs += self.qf(p_float).sum()
                norm_precs += self.qf(norm_p_float).sum()
                num_precs += m.bias_s.numel()

        if num_precs == 0: return "Network is in full precision"
        return "Network precision with {}: {} / {}".format(self.qf.__name__, precs/num_precs, norm_precs/num_precs)

    def print_precs_quantize(self):
        precs = 0.
        num_precs = 0.
        for m in self.mixed_layers:
            _, p = Quantize.apply(m.weight, m.weight_s, m.qf)
            precs += p.sum().item()
            num_precs += p.numel()

            if m.bias is not None:
                _, bp = Quantize.apply(m.bias, m.bias_s, m.qf)
                precs += bp.sum().item()
                num_precs += bp.numel()

        return "Network ACTUAL precision with {}: {}".format(self.qf.__name__, precs/num_precs)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, NoisyConv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class PreActBlock(nn.Module):
    def __init__(self, inplanes, planes, stride, p_init, first=False, spm='None'):
        super(PreActBlock, self).__init__()
        self.shortcut = None
        if first:
            self.preact = nn.Identity()
            self.shortcut = Shortcut(inplanes, planes, stride)
        else:
            self.preact = nn.Sequential(nn.BatchNorm2d(inplanes), nn.ReLU())

        self.conv1 = conv(inplanes, planes, 3, stride, p_init, fp=False, spm=spm)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU()
        self.conv2 = conv(planes, planes, 3, 1, p_init, fp=False, spm=spm)
        self.bn2 = nn.BatchNorm2d(planes)
        self.outplanes = planes

    def forward(self, x):
        residual = x
        out = self.preact(x)
        out = self.relu1(self.bn1(self.conv1(out)))
        out = self.bn2(self.conv2(out))
        if self.shortcut is not None: residual = self.shortcut(residual)
        return out + residual

class PreActBottleneckBlock(nn.Module):
    def __init__(self, inplanes, planes, stride, p_init, first=False, spm='None'):
        super(PreActBottleneckBlock, self).__init__()
        self.shortcut = None
        if first:
            self.preact = nn.Identity()
            self.shortcut = Shortcut(inplanes, 4*planes, stride)
        else:
            self.preact = nn.Sequential(nn.BatchNorm2d(inplanes), nn.ReLU(inplace=True))

        self.conv1 = conv(inplanes, planes, 1, 1, p_init, fp=False, spm=spm)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv(planes, planes, 3, stride, p_init, fp=False, spm=spm)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = conv(planes, 4*planes, 1, 1, p_init, fp=False, spm=spm)
        self.bn3 = nn.BatchNorm2d(4*planes)
        self.outplanes = 4*planes

    def forward(self, x):
        residual = x
        out = self.preact(x)
        out = self.relu1(self.bn1(self.conv1(out)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.shortcut is not None: residual = self.shortcut(residual)
        return out + residual

class PreResNet(MixedNet):
    def __init__(self, block_sizes, bottleneck, mode='quant', qf=torch.floor, p_init=32, spm='None'):
        super(PreResNet, self).__init__(mode, qf, p_init)
        self.p_init = p_init
        self.spm = spm

        if bottleneck:
            self.block_type = PreActBottleneckBlock
        else:
            self.block_type = PreActBlock

        self.pre = nn.Sequential(conv(3, 64, 7, 2, p_init, fp=True, spm=self.spm), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.planes = 64
        self.stage1 = self.make_stage(64, block_sizes[0], 1, spm=self.spm)
        self.stage2 = self.make_stage(128, block_sizes[1], 2, spm=self.spm)
        self.stage3 = self.make_stage(256, block_sizes[2], 2, spm=self.spm)
        self.stage4 = self.make_stage(512, block_sizes[3], 2, spm=self.spm)
        self.pool = nn.AvgPool2d(8)
        self.fc = linear(self.planes, 1000, p_init, fp=True, spm=self.spm )

        self.mixed_layers = [m for m in self.modules() if isinstance(m, NoisyConv2d) or isinstance(m, NoisyLinear)]
        self.mixed_layers[0].firstLayer = True
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def make_stage(self, planes, blocks, stride, spm='None'):
        layers = [self.block_type(self.planes, planes, stride, self.p_init, first=True, spm=self.spm)]
        self.planes = layers[-1].outplanes
        for i in range(1, blocks):
            layers.append(self.block_type(self.planes, planes, 1, self.p_init, spm=self.spm))
            self.planes = layers[-1].outplanes
        layers.append(nn.BatchNorm2d(self.planes))
        layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        # print("forward done", self.fc.p == None)

        return x