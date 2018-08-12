#coding:utf-8
# author: Gaosheng Ren, Call me Marks will be cool.
# 

from mxnet import gluon
import mxnet.gluon.nn as nn
from mxnet import image
import mxnet as mx
from mxnet import nd
import numpy as np
from mxnet.gluon.block import HybridBlock



# 通道shuffle
class shuffle_channels(nn.HybridBlock):
    """
    ShuffleNet channel shuffle Block.
    """
    def __init__(self, groups=3, **kwargs):
        super(shuffle_channels, self).__init__()
        self.groups = groups

    def hybrid_forward(self, F, x):
        data = F.reshape(x, shape=(0, -4, self.groups, -1, -2))
        data = F.swapaxes(data, 1, 2)
        data = F.reshape(data, shape=(0, -3, -2))
        return data


# ShuffleNet stride=1
class ShuffleNetUnitA(HybridBlock):
    """ShuffleNet unit for stride=1"""
    def __init__(self, in_channels, out_channels, groups=3, **kwargs):
        super(ShuffleNetUnitA, self).__init__()
        assert in_channels == out_channels
        assert out_channels % 4 == 0
        bottleneck_channels = out_channels // 4
        self.groups = groups
        with self.name_scope():
            self.group_conv1 = nn.Conv2D(channels=bottleneck_channels, kernel_size=1, strides=1, padding=0, groups=groups)        
            self.bn2 = nn.BatchNorm()
            self.shuffle_channels = shuffle_channels(groups=self.groups)
            #depthwise
            self.depthwise_conv3 = nn.Conv2D(channels=bottleneck_channels,kernel_size=3, strides=1, padding=1, groups=bottleneck_channels, use_bias=False)
            self.bn4 = nn.BatchNorm()
            self.group_conv5 = nn.Conv2D(channels=out_channels, kernel_size=1, strides=1, padding=0, groups=groups)
            self.bn6 = nn.BatchNorm()


    def hybrid_forward(self, F, x):
        out = self.group_conv1(x)
        out = F.relu(self.bn2(out))
        out = self.shuffle_channels(out)
        out = self.depthwise_conv3(out)
        out = self.bn4(out)
        out = self.group_conv5(out)
        out = self.bn6(out)
        out = F.relu(F.elemwise_add(x,out))# or relu(x+out)
        return out


