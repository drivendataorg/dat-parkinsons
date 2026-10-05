import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial

__all__ = ['SelfAttentionResNeXt', 'resnext50_attn', 'resnext101_attn', 'resnext152_attn', 'resnext200_attn', 'resnext269_attn']


def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out

class NonLocalBlock3D(nn.Module):
    """
    Self-Attention / Non-Local module for 3D feature maps (Video / Volumetric data).
    Calculates self-attention across Temporal, Spatial-H, and Spatial-W dimensions.
    """
    def __init__(self, in_channels, inter_channels=None, sub_sample=True):
        super(NonLocalBlock3D, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        # Query, Key, Value projections
        self.g = nn.Conv3d(in_channels, self.inter_channels, kernel_size=1)
        self.theta = nn.Conv3d(in_channels, self.inter_channels, kernel_size=1)
        self.phi = nn.Conv3d(in_channels, self.inter_channels, kernel_size=1)

        # Output projection
        self.W = nn.Sequential(
            nn.Conv3d(self.inter_channels, in_channels, kernel_size=1),
            nn.BatchNorm3d(in_channels)
        )
        
        # Zero-initialize the last BN layer so the module acts as an identity at start
        nn.init.constant_(self.W[1].weight, 0)
        nn.init.constant_(self.W[1].bias, 0)

        # MaxPool for Key and Value to reduce self-attention memory cost
        if sub_sample:
            self.max_pool = nn.MaxPool3d(kernel_size=(1, 2, 2))
            self.phi = nn.Sequential(self.phi, self.max_pool)
            self.g = nn.Sequential(self.g, self.max_pool)

    def forward(self, x):
        batch_size = x.size(0)

        # Value: [B, C_inter, T*H*W]
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        # Query: [B, C_inter, T*H*W]
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)

        # Key: [B, C_inter, T*H*W]
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)

        # Self-Attention Map: [B, N, N] where N = T*H*W
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        # Aggregate Values using Attention weights
        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        
        # Out projection + Residual connection
        W_y = self.W(y)
        out = W_y + x

        return out

class ResNeXtBottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, cardinality, stride=1,
                 downsample=None):
        super(ResNeXtBottleneck, self).__init__()
        mid_planes = cardinality * int(planes / 32)
        self.conv1 = nn.Conv3d(inplanes, mid_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(mid_planes)
        self.conv2 = nn.Conv3d(
            mid_planes,
            mid_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=cardinality,
            bias=False)
        self.bn2 = nn.BatchNorm3d(mid_planes)
        self.conv3 = nn.Conv3d(
            mid_planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SelfAttentionResNeXt(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 sample_size,
                 sample_duration,
                 shortcut_type='B',
                 cardinality=32,
                 num_classes=400,
                 non_local_layers=[False, True, True, False]): 
        """
        non_local_layers: A list of booleans specifying whether to add self-attention
                          at the end of layer1, layer2, layer3, and layer4.
        """
        self.inplanes = 64
        super(SelfAttentionResNeXt, self).__init__()
        
        self.conv1 = nn.Conv3d(
            1, 64, kernel_size=7, stride=(1, 2, 2), padding=(3, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        
        # Layer 1
        self.layer1 = self._make_layer(
            block, 128, layers[0], shortcut_type, cardinality, stride=1,
            add_non_local=non_local_layers[0])
        
        # Layer 2 (Self-attention inserted here by default)
        self.layer2 = self._make_layer(
            block, 256, layers[1], shortcut_type, cardinality, stride=2,
            add_non_local=non_local_layers[1])
        
        # Layer 3 (Self-attention inserted here by default)
        self.layer3 = self._make_layer(
            block, 512, layers[2], shortcut_type, cardinality, stride=2,
            add_non_local=non_local_layers[2])
        
        # Layer 4
        self.layer4 = self._make_layer(
            block, 1024, layers[3], shortcut_type, cardinality, stride=2,
            add_non_local=non_local_layers[3])
        
        last_duration = int(math.ceil(sample_duration / 16))
        last_size = int(math.ceil(sample_size / 32))
        self.avgpool = nn.AvgPool3d((last_duration, last_size, last_size), stride=1)
        self.fc = nn.Linear(cardinality * 32 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, shortcut_type, cardinality, stride=1, add_non_local=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, cardinality, stride, downsample))
        self.inplanes = planes * block.expansion
        
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cardinality))

        # Attach self-attention block after stage convolutions
        if add_non_local:
            layers.append(NonLocalBlock3D(self.inplanes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x



def resnext50_attn(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = SelfAttentionResNeXt(ResNeXtBottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnext101_attn(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = SelfAttentionResNeXt(ResNeXtBottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnext152_attn(**kwargs):
    """Constructs a ResNet-152 model.
    """
    model = SelfAttentionResNeXt(ResNeXtBottleneck, [3, 8, 36, 3], **kwargs)
    return model


def resnext200_attn(**kwargs):
    """Constructs a ResNeXt-200 model."""
    model = SelfAttentionResNeXt(ResNeXtBottleneck, [3, 24, 36, 3], **kwargs)
    return model


def resnext269_attn(**kwargs):
    """Constructs a ResNeXt-269 model."""
    model = SelfAttentionResNeXt(ResNeXtBottleneck, [3, 30, 52, 3], **kwargs)
    return model