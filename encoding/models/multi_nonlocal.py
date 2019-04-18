import math
import torch
import torch.utils.model_zoo as model_zoo
import torch.nn as nn
import torch.nn.functional as F

from ..nn import ConcurrentModule, SyncBatchNorm
from ..nn import GlobalAvgPool2d


class Interpolate(nn.Module):
    def __init__(self, scale, mode):
        super(Interpolate, self).__init__()
        self.interp = F.interpolate
        self.scale = scale
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale, mode='nearest')
        return x


class NonLocal(nn.Module):
    """Non-local Block"""

    def __init__(self, channel_in, channel_reduced, subsample=True, use_bn=True):
        super(NonLocal, self).__init__()
        if subsample:
            # Reduce dim
            self.phy = nn.Sequential(
                nn.Conv2d(channel_in, channel_reduced, kernel_size=1, stride=1),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
            # Transform
            self.g = nn.Sequential(
                nn.Conv2d(channel_in, channel_reduced, kernel_size=1, stride=1),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
        else:
            # Reduce dim
            self.phy = nn.Conv2d(channel_in, channel_reduced, kernel_size=1, stride=1)
            # Transform
            self.g = nn.Conv2d(channel_in, channel_reduced, kernel_size=1, stride=1)

        # Reduce dim
        self.theta = nn.Conv2d(channel_in, channel_reduced, kernel_size=1, stride=1)

        if use_bn:
            # Extend
            self.extend = nn.Sequential(
                nn.Conv2d(channel_reduced, channel_in, kernel_size=1, stride=1),
                SyncBatchNorm(channel_in),
            )
        else:
            self.extend = nn.Conv2d(channel_reduced, channel_in, kernel_size=1, stride=1)

    def forward(self, x):
        H, W = x.size(2), x.size(3)

        # Reduce dim and transform
        x_theta = self.theta(x)                # [batch, channel_reduced, H, W]
        x_phy = self.phy(x)                    # [batch, channel_reduced, H / 2, W / 2]
        x_g = self.g(x)                        # [batch, channel_reduced, H / 2, W / 2]

        x_theta = x_theta.view(x_theta.size(0), x_theta.size(1), -1)   # [batch, channel_reduced, H * W]
        x_theta = torch.transpose(x_theta, 1, 2)                       # [batch, H * W, channel_reduced]

        x_phy = x_phy.view(x_phy.size(0), x_phy.size(1), -1)           # [batch, channel_reduced, H * W / 4]

        # Similarity matrix between every spatial location pair
        x_sim = torch.bmm(x_theta, x_phy)                              # [batch, H * W, H * W / 4]
        x_sim /= x_sim.size(-1)  # scaling by 1/N
        x_sim = torch.transpose(x_sim, 1, 2)                           # [batch, H * W / 4, H * W]

        x_g = x_g.view(x_g.size(0), x_g.size(1), -1)                   # [batch, channel_reduced, H * W / 4]

        # Self-attention
        x_out = torch.bmm(x_g, x_sim)                                  # [batch, channel_reduced, H * W]
        x_out = x_out.view(x_out.size(0), x_out.size(1), H, W)         # [batch, channel_reduced, H, W]

        # Extend dim
        x_out = self.extend(x_out)                                     # [batch, channel_in, H, W]

        return x_out


class ConcatFusion(nn.Module):
    def __init__(self, channel_in, group=2, H=None, W=None):
        super(ConcatFusion, self).__init__()
        print("Concat fusion")
        self.group_num = group

        self.fusion = nn.Conv2d(channel_in*group, channel_in, kernel_size=3, stride=1, padding=1)
        self.bn = SyncBatchNorm(channel_in)

    def forward(self, x):
        x_fuse = torch.cat(x, dim=1)    # [batch, channel_in*group, H, W]
        x_fuse = self.fusion(x_fuse)       # [batch, channel_in, H, W]
        x_fuse = self.bn(x_fuse)
        return x_fuse


class ChannelSelectiveFusion(nn.Module):
    def __init__(self, channel_in, group=2, ratio=4):
        super(ChannelSelectiveFusion, self).__init__()
        print("Channel Selective fusion")
        self.group_num = group
        self.r = ratio
        self.channel_in = channel_in

        self.fuse = GlobalAvgPool2d()
        self.reduce = nn.Linear(channel_in, channel_in // self.r)
        self.relu = nn.ReLU(inplace=True)
        self.attention_layer = nn.Linear(channel_in // self.r, channel_in * group)

        self.softmax = nn.Softmax(dim=2)
        self.bn = SyncBatchNorm(channel_in)

    def forward(self, x):
        # x: (group, [batch, channel_in, H, W])
        x_fuse = torch.zeros_like(x[0])  # [batch, channel_in, H, W]
        for i in range(self.group_num):
            x_fuse += x[i]

        # Squeeze
        x_fuse = self.fuse(x_fuse)  # [batch, channel_in]
        x_fuse = self.reduce(x_fuse)  # [batch, channel_in // r]
        x_fuse = self.relu(x_fuse)

        # Excite
        x_atten_all = self.attention_layer(x_fuse)   # [batch, channel_in * group]
        x_atten_all = x_atten_all.view(-1, self.channel_in, self.group_num)     # [batch, channel_in, group]

        # Attention
        x_atten_all = self.softmax(x_atten_all)
        x_atten_all = torch.split(x_atten_all, 1, dim=2)  # (group, [batch, channel_in, 1])
        x_atten_all = list(x_atten_all)

        x_out = torch.zeros_like(x[0])
        for i in range(self.group_num):
            # [group, [batch, channel_in, 1, 1]]
            x_atten_all[i] = x_atten_all[i].view(x_atten_all[i].size(0), x_atten_all[i].size(1), 1, 1)
            x_out += x[i] * x_atten_all[i]    # [batch, channel_in, H, W]

        x_out = self.bn(x_out)

        return x_out


class MultiNonLocal(nn.Module):
    def __init__(self, channel_in, channel_reduced, branches=[1]):
        super(MultiNonLocal, self).__init__()

        print("MultiNonLocal Module branches: ", branches)

        self.nonlocal_layers = nn.ModuleList()
        for i, ratio in enumerate(branches):
            self.nonlocal_layers.append(nn.Sequential())
            if ratio > 1:
                self.nonlocal_layers[i].add_module("maxpool_{}x".format(ratio), nn.MaxPool2d(ratio, ratio))
            self.nonlocal_layers[i].add_module('nonlocal_{}x'.format(ratio), NonLocal(channel_in, channel_reduced, subsample=False))
            self.nonlocal_layers[i].add_module('relu_{}x'.format(ratio), nn.ReLU(inplace=True))
            if ratio > 1:
                self.nonlocal_layers[i].add_module('interp_{}x'.format(ratio), Interpolate(ratio, mode='nearest'))
                self.nonlocal_layers[i].add_module('conv_tail_{}x'.format(ratio), nn.Sequential(
                    nn.Conv2d(channel_in, channel_in, kernel_size=1, stride=1),
                    SyncBatchNorm(channel_in),
                    nn.ReLU(inplace=True)
                ))
        if len(branches) > 1:
            self.fusion = ChannelSelectiveFusion(channel_in, group=len(branches))

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        multi_x = []
        for layer in self.nonlocal_layers:
            multi_x.append(layer(x))     # [group, [batch, channel, H, W]]

        # Align H and W of each branch feature for multi-scale testing
        if len(multi_x) > 1:
            for i, _ in enumerate(multi_x):
                if multi_x[i].size() != x.size():
                    multi_x[i] = F.interpolate(multi_x[i], size=x.size()[2:], mode='nearest')

        if len(multi_x) > 1:
            fused_x = self.fusion(multi_x)    # [batch, channel, H, W]
        else:
            fused_x = multi_x[0]

        x_out = identity + fused_x
        x_out = self.relu(x_out)

        return x_out
