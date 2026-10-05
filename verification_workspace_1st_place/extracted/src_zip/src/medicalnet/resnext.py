import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial

__all__ = [
    'ResNeXt', 
    'resnext50', 'resnext101', 'resnext152', 'resnext200', 'resnext269', 
    'seresnext50', 'seresnext101', 'seresnext152',
    'skresnext50', 'skresnext101', 'skresnext152'
]


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


class ResNeXt(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 sample_size,
                 sample_duration,
                 shortcut_type='B',
                 cardinality=32,
                 num_classes=400):
        self.inplanes = 64
        super(ResNeXt, self).__init__()
        self.conv1 = nn.Conv3d(
            1,
            64,
            kernel_size=7,
            stride=(1, 2, 2),
            padding=(3, 3, 3),
            bias=False)
        #self.conv1 = nn.Conv3d(
        #    3,
        #    64,
        #    kernel_size=(3,7,7),
        #    stride=(1, 2, 2),
        #    padding=(1, 3, 3),
        #    bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 128, layers[0], shortcut_type,
                                       cardinality)
        self.layer2 = self._make_layer(
            block, 256, layers[1], shortcut_type, cardinality, stride=2)
        self.layer3 = self._make_layer(
            block, 512, layers[2], shortcut_type, cardinality, stride=2)
        self.layer4 = self._make_layer(
            block, 1024, layers[3], shortcut_type, cardinality, stride=2)
        last_duration = int(math.ceil(sample_duration / 16))
        #last_duration = 1
        last_size = int(math.ceil(sample_size / 32))
        self.avgpool = nn.AvgPool3d(
            (last_duration, last_size, last_size), stride=1)
        self.fc = nn.Linear(cardinality * 32 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self,
                    block,
                    planes,
                    blocks,
                    shortcut_type,
                    cardinality,
                    stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(self.inplanes, planes, cardinality, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cardinality))

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

# ==========================================================================
# SE-Resnext blocks
# ==========================================================================
class SEBlock3D(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock3D, self).__init__()
        # Global Average Pooling over Depth, Height, Width
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        # Squeeze
        y = self.avg_pool(x).view(b, c)
        # Excitation
        y = self.fc(y).view(b, c, 1, 1, 1)
        # Scale input features
        return x * y.expand_as(x)


class SEResNeXtBottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, cardinality, stride=1, downsample=None, reduction=16):
        super(SEResNeXtBottleneck, self).__init__()
        mid_planes = cardinality * int(planes / 32)
        
        self.conv1 = nn.Conv3d(inplanes, mid_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(mid_planes)
        
        self.conv2 = nn.Conv3d(
            mid_planes, mid_planes, kernel_size=3, stride=stride,
            padding=1, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm3d(mid_planes)
        
        self.conv3 = nn.Conv3d(
            mid_planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        
        self.relu = nn.ReLU(inplace=True)
        
        # --- ATTENTION MODULE ---
        self.se = SEBlock3D(planes * self.expansion, reduction=reduction)
        
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

        # Apply SE Attention before residual connection
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

# ==========================================================================
# SK-Resnext blocks
# ==========================================================================
class SKConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, M=2, G=32, r=16, L=32):
        """
        3D Selective Kernel Convolution Module
        M: Number of parallel branches (default=2: kernel 3x3x3 and dilated 3x3x3)
        G: Cardinality / Groups
        r: Reduction ratio for attention vector
        L: Minimum dimension for attention vector
        """
        super(SKConv3D, self).__init__()
        self.M = M
        self.out_channels = out_channels
        
        # Branch 1: Standard 3x3x3 Grouped Conv (effective RF = 3x3x3)
        # Branch 2: Dilated 3x3x3 Grouped Conv (effective RF = 5x5x5)
        self.convs = nn.ModuleList([])
        for i in range(M):
            self.convs.append(nn.Sequential(
                nn.Conv3d(
                    in_channels, out_channels, kernel_size=3, stride=stride,
                    padding=1 + i, dilation=1 + i, groups=G, bias=False
                ),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True)
            ))
            
        # Attention vector dimension
        d = max(int(out_channels / r), L)
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(out_channels, d, bias=False),
            nn.LayerNorm(d),  # <-- Use LayerNorm instead of BatchNorm1d
            nn.ReLU(inplace=True)
        )
        self.fcs = nn.ModuleList([nn.Linear(d, out_channels, bias=False) for _ in range(M)])
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.size(0)
        
        # 1. SPLIT: Compute features across multiple branches
        feats = [conv(x) for conv in self.convs]             # List of M Tensors: [B, C, D, H, W]
        U = torch.sum(torch.stack(feats, dim=0), dim=0)   # Element-wise summation: [B, C, D, H, W]
        
        # 2. FUSE: Squeeze global spatial/temporal context
        S = self.global_pool(U).view(batch_size, -1)      # Global pooling: [B, C]
        Z = self.fc(S)                                    # Compact vector: [B, d]
        
        # 3. SELECT: Generate Softmax weights across branches
        weights = [fc(Z) for fc in self.fcs]              # List of M Tensors: [B, C]
        weights = torch.stack(weights, dim=1)             # [B, M, C]
        weights = self.softmax(weights)                   # Softmax over M branches
        
        # Reshape for 3D broadcasting: [B, M, C, 1, 1, 1]
        weights = weights.view(batch_size, self.M, self.out_channels, 1, 1, 1)
        
        # Weighted combination of branch features
        V = torch.sum(torch.stack(feats, dim=1) * weights, dim=1)
        return V


class SKResNeXtBottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, cardinality=32, stride=1, downsample=None, reduction=16):
        super(SKResNeXtBottleneck, self).__init__()
        mid_planes = cardinality * int(planes / 32)

        # 1x1x1 bottleneck reduction
        self.conv1 = nn.Conv3d(inplanes, mid_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(mid_planes)

        # Selective Kernel 3x3x3 module replacing standard Conv2
        self.sk_conv = SKConv3D(
            in_channels=mid_planes,
            out_channels=mid_planes,
            stride=stride,
            M=2,
            G=cardinality,
            r=reduction
        )

        # 1x1x1 bottleneck expansion
        self.conv3 = nn.Conv3d(mid_planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # SK-Conv replaces standard 3x3x3 conv + bn + relu
        out = self.sk_conv(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


# ==========================================================================
# CBAM-Resnext blocks
# ==========================================================================

class ChannelAttention3D(nn.Module):
    def __init__(self, in_planes, reduction=16):
        super(ChannelAttention3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        
        # Shared MLP for both AvgPool and MaxPool
        self.fc = nn.Sequential(
            nn.Conv3d(in_planes, in_planes // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_planes // reduction, in_planes, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention3D(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention3D, self).__init__()
        # Kernel size must be odd for proper padding
        padding = kernel_size // 2
        self.conv = nn.Conv3d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Channel reduction via global average and max pooling
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate along channel dimension [B, 2, D, H, W]
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out)


class CBAM3D(nn.Module):
    def __init__(self, planes, reduction=16, kernel_size=7):
        super(CBAM3D, self).__init__()
        self.ca = ChannelAttention3D(planes, reduction)
        self.sa = SpatialAttention3D(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)       # Apply channel attention
        x = x * self.sa(x)       # Apply spatial/temporal attention
        return x


class CBAMResNeXtBottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, cardinality, stride=1, downsample=None, reduction=16):
        super(CBAMResNeXtBottleneck, self).__init__()
        mid_planes = cardinality * int(planes / 32)
        
        self.conv1 = nn.Conv3d(inplanes, mid_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(mid_planes)
        
        self.conv2 = nn.Conv3d(
            mid_planes, mid_planes, kernel_size=3, stride=stride,
            padding=1, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm3d(mid_planes)
        
        self.conv3 = nn.Conv3d(
            mid_planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        
        self.relu = nn.ReLU(inplace=True)
        
        # --- CBAM ATTENTION MODULE ---
        self.cbam = CBAM3D(planes * self.expansion, reduction=reduction)
        
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

        # Apply CBAM Attention (Channel + Spatial) before residual connection
        out = self.cbam(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
        
### ===================================


def resnext50(**kwargs):
    """Constructs a ResNet-50 model."""
    model = ResNeXt(ResNeXtBottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnext101(**kwargs):
    """Constructs a ResNet-101 model."""
    model = ResNeXt(ResNeXtBottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnext152(**kwargs):
    """Constructs a ResNet-152 model."""
    model = ResNeXt(ResNeXtBottleneck, [3, 8, 36, 3], **kwargs)
    return model


def resnext200(**kwargs):
    """Constructs a ResNeXt-200 model."""
    model = ResNeXt(ResNeXtBottleneck, [3, 24, 36, 3], **kwargs)
    return model


def resnext269(**kwargs):
    """Constructs a ResNeXt-269 model."""
    model = ResNeXt(ResNeXtBottleneck, [3, 30, 52, 3], **kwargs)
    return model


# =========================== SE-RESNEXT =======================================
def seresnext50(**kwargs):
    """Constructs a SE-ResNet-50 model."""
    model = ResNeXt(SEResNeXtBottleneck, [3, 4, 6, 3], **kwargs)
    return model


def seresnext101(**kwargs):
    """Constructs a SE-ResNet-50 model."""
    model = ResNeXt(SEResNeXtBottleneck, [3, 4, 23, 3], **kwargs)
    return model


def seresnext152(**kwargs):
    """Constructs a ResNet-152 model."""
    model = ResNeXt(SEResNeXtBottleneck, [3, 8, 36, 3], **kwargs)
    return model


# =========================== SK-RESNEXT =======================================
def skresnext50(**kwargs):
    """Constructs a SE-ResNet-50 model."""
    model = ResNeXt(SKResNeXtBottleneck, [3, 4, 6, 3], **kwargs)
    return model


def skresnext101(**kwargs):
    """Constructs a SE-ResNet-101 model."""
    model = ResNeXt(SKResNeXtBottleneck, [3, 4, 23, 3], **kwargs)
    return model


def skresnext152(**kwargs):
    """Constructs a ResNet-152 model."""
    model = ResNeXt(SKResNeXtBottleneck, [3, 8, 36, 3], **kwargs)
    return model


# =========================== CBAM-RESNEXT =======================================
def cbamresnext50(**kwargs):
    """Constructs a CBAM-ResNet-50 model."""
    model = ResNeXt(CBAMResNeXtBottleneck, [3, 4, 6, 3], **kwargs)
    return model

def cbamresnext101(**kwargs):
    """Constructs a CBAM-ResNet-101 model."""
    model = ResNeXt(CBAMResNeXtBottleneck, [3, 4, 23, 3], **kwargs)
    return model

def cbamresnext152(**kwargs):
    """Constructs a CBAM-ResNet-152 model."""
    model = ResNeXt(CBAMResNeXtBottleneck, [3, 8, 36, 3], **kwargs)
    return model