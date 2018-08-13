#coding:utf-8
# author: Gaosheng Ren, Call me Marks will be cool.
# iamrgs@foxmail.com

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

# strike=2的基本单元
class ShuffleNetUnitB(HybridBlock):
    """ShuffleNet unit for stride=2"""
    def __init__(self, in_channels, out_channels, groups=3, **kwargs):
        super(ShuffleNetUnitB, self).__init__(**kwargs)
        out_channels -= in_channels
        assert out_channels % 4 == 0
        bottleneck_channels = out_channels // 4
        self.groups = groups
        with self.name_scope():
            self.group_conv1 = nn.Conv2D(bottleneck_channels, kernel_size=1, strides=1, padding=0, groups=groups)
            self.bn2 = nn.BatchNorm()
            self.shuffle_channels = shuffle_channels(groups=self.groups)
            self.depthwise_conv3 = nn.Conv2D(bottleneck_channels, kernel_size=3, strides=2, padding=1, groups=bottleneck_channels, use_bias=False)
            self.bn4 = nn.BatchNorm()
            self.group_conv5 = nn.Conv2D(out_channels, kernel_size=1, strides=1, groups=groups)
            self.bn6 = nn.BatchNorm()
            self.avg_pool = nn.AvgPool2D(pool_size=(3, 3), strides=(2,2), padding=1)

    def hybrid_forward(self, F, x):
        out = self.group_conv1(x)
        out = F.relu(self.bn2(out))
        out = self.shuffle_channels(out)
        out = self.depthwise_conv3(out)
        out = self.bn4(out)
        out = self.group_conv5(out)
        out = self.bn6(out)
        x = self.avg_pool(x)
        out = F.relu(F.concat(x, out, dim=1))# channel 叠加
        return out

# relu6 实现
class RELU6(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(RELU6, self).__init__(**kwargs)

    def hybrid_forward(self, F, x):
        return F.clip(x, 0, 6, name="fwd")

# g=3的ShuffleNet的实现
class ShuffleNet(HybridBlock):
    """ShuffleNet for groups=3"""
    def __init__(self, groups=3, num_classes=10, **kwargs):
        super(ShuffleNet, self).__init__(**kwargs)
        
        with self.name_scope():
#             self.conv1 = nn.Conv2D(channels=24, kernel_size=3, stride=2, padding=1)
#             self.max_pool2d = nn.MaxPool2D(pool_size=(3, 3), strides=(2,2), padding=1)
            
            self.features = nn.HybridSequential(prefix='shufflenet_')
            with self.features.name_scope():
                
                self.features.add(nn.Conv2D(channels=24, kernel_size=3, strides=2, padding=1))
                self.features.add(nn.MaxPool2D(pool_size=(3, 3), strides=(2,2), padding=1))

                #stage2_seq
                self.features.add(ShuffleNetUnitB(24,240, groups=3))#24,240
                for i in range(3):
                    self.features.add(ShuffleNetUnitA(240, 240, groups=3)) #240, 240

                #stage3_seq 
                self.features.add(ShuffleNetUnitB(240, 480, groups=3))#240, 480
                for i in range(7):
                    self.features.add(ShuffleNetUnitA(480, 480, groups=3))#480, 480, 

                #stage4_seq
                self.features.add(ShuffleNetUnitB(480, 960, groups=3))#480, 960,
                for i in range(3):
                    self.features.add(ShuffleNetUnitA(960, 960, groups=3))#960, 960

                self.features.add(nn.GlobalAvgPool2D())
                self.features.add(nn.Dense(num_classes))#, activation="linear" means a(x) = x
#              nn.Activation(activation='relu')
#             self.avg_pool = nn.AvgPool2D(pool_size=(7, 7))                
#             self.fc = nn.Dense(num_classes, activation="linear")

    def hybrid_forward(self, F, x):
#         net = self.conv1(x)
#         net = self.max_pool2d(net)
        net = self.features(x)
#         net = self.avg_pool(net)
#         net = F.avg_pool2d(net, 7)
#         net = net.view(net.size(0), -1)
#         net = self.fc(net)
        logits = F.softmax(net)
        return logits
