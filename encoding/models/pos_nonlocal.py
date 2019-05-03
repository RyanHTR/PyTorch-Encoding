import math
import torch
import torch.utils.model_zoo as model_zoo
import torch.nn as nn
import torch.nn.functional as F

from ..nn import ConcurrentModule, SyncBatchNorm
from ..nn import GlobalAvgPool2d


class Partition2d(nn.Module):
    pass

class Recover2d(nn.Module):
    pass

def scaler_norm(x):
    max = x.max()
    min = x.min()
    return (x - min) / (max - min)


def get_pos_sim(batch_size, height, width):
    x = torch.arange(width).cuda().float().view(1, width)    # [1, w]
    x = x.repeat(height, 1).view(-1)                    # [hw]
    x_dis = (x[:, None] - x[None, :]).abs()            # [hw, hw]

    y = torch.arange(height).cuda().float().view(height, 1)
    y = y.repeat(1, width).view(-1)
    y_dis = (y[:, None] - y[None, :]).abs()

    A = x_dis ** 2 + y_dis ** 2
    A = scaler_norm(A)

    A = A.unsqueeze(0)
    A = A.repeat(batch_size, 1, 1)    # [batch_size, h*w, h*w]

    return A


class Interpolate(nn.Module):
    def __init__(self, scale, mode):
        super(Interpolate, self).__init__()
        self.interp = F.interpolate
        self.scale = scale
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale, mode='nearest')
        return x


class NonLocalPos(nn.Module):
    """Non-local Block"""

    def __init__(self, channel_in, channel_reduced, subsample=True, use_bn=True):
        super(NonLocalPos, self).__init__()
        # Transform
        self.g = nn.Conv2d(channel_in, channel_reduced, kernel_size=1, stride=1)

        if use_bn:
            # Extend
            self.extend = nn.Sequential(
                nn.Conv2d(channel_reduced, channel_in, kernel_size=1, stride=1),
                SyncBatchNorm(channel_in),
            )
        else:
            self.extend = nn.Conv2d(channel_reduced, channel_in, kernel_size=1, stride=1)

    def forward(self, x):
        batch_size, H, W = x.size(0), x.size(2), x.size(3)

        # get similarity
        x_sim = get_pos_sim(batch_size, H, W)     #[batch, HW, HW]

        # Reduce dim and transform
        x_g = self.g(x)                        # [batch, channel_reduced, H, W]
        x_g = x_g.view(x_g.size(0), x_g.size(1), -1)                   # [batch, channel_reduced, H * W]

        # Self-attention
        x_out = torch.bmm(x_g, x_sim)                                  # [batch, channel_reduced, H * W]
        x_out = x_out.view(x_out.size(0), x_out.size(1), H, W)         # [batch, channel_reduced, H, W]

        # Extend dim
        x_out = self.extend(x_out)                                     # [batch, channel_in, H, W]

        return x_out


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


class AttentionScaleFusion(nn.Module):
    """Attention to scale"""
    def __init__(self, channel_in, group=2, ratio=4):
        super(AttentionScaleFusion, self).__init__()
        print("Selective fusion using attention to scale")
        self.group_num = group
        self.r = ratio

        # attention module
        channel_inter = channel_in // ratio
        # self.attention_layer = nn.Sequential(nn.Conv2d(channel_in*group, channel_inter, 3, stride=1, padding=1),
        #                                      SyncBatchNorm(channel_inter),
        #                                      nn.ReLU(inplace=True),
        #                                      nn.Conv2d(channel_inter, group, 1))
        self.attention_layer = nn.Sequential(nn.Conv2d(channel_in, channel_inter, 3, stride=1, padding=1),
                                             SyncBatchNorm(channel_inter),
                                             nn.ReLU(inplace=True),
                                             nn.Conv2d(channel_inter, group, 1))

        self.softmax = nn.Softmax(dim=1)
        self.bn = SyncBatchNorm(channel_in)

    def forward(self, x):
        # x: (group, [batch, channel_in, H, W])
        x_fuse = torch.zeros_like(x[0])            # [batch, channel_in, H, W]
        for x_i in x:
            x_fuse = x_fuse + x_i
        # x_fuse = torch.cat(x, dim=1)          # [batch, channel_in * group, H, W]

        # get attention map
        attn = self.attention_layer(x_fuse)   # [batch, group, H, W]
        attn = self.softmax(attn)
        attn = torch.split(attn, 1, dim=1)  # (group, [batch, 1, H, W])

        # weighted sum
        x_out = torch.zeros_like(x[0])
        for i in range(self.group_num):
            # [group, [batch, 1, H, W]]
            x_out += x[i] * attn[i]    # [batch, channel_in, H, W]

        x_out = self.bn(x_out)

        return x_out


class MultiNonLocal(nn.Module):
    def __init__(self, channel_in, channel_reduced, branches=[1]):
        super(MultiNonLocal, self).__init__()

        print("MultiNonLocal Module Regions: ", branches)

        self.nonlocal_layers = nn.ModuleList()
        for i, ratio in enumerate(branches):
            self.nonlocal_layers.append(nn.Sequential())
            if ratio > 1:
                self.nonlocal_layers[i].add_module("partition_{}x".format(ratio), Partition2d(ratio=ratio))
            self.nonlocal_layers[i].add_module('nonlocal_pos_{}x'.format(ratio), NonLocalPos(channel_in, channel_reduced, subsample=False))
            self.nonlocal_layers[i].add_module('relu_{}x'.format(ratio), nn.ReLU(inplace=True))
            if ratio > 1:
                self.nonlocal_layers[i].add_module('recover_{}x'.format(ratio), Recover2d(ratio=ratio))
        if len(branches) > 1:
            self.fusion = AttentionScaleFusion(channel_in, group=len(branches))

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
