import torch
import torch.nn as nn
from types import SimpleNamespace
from torchinfo import summary
import sys


from .medicalnet.model import generate_model
from .medicalnet.setting import parse_opts
from .mednext.create_mednext_v1 import create_mednext_v1


# ===========================================================================
### MedicalNet
# ===========================================================================
class MedicalNetClassifier(nn.Module):
    def __init__(self, medicalnet):
        super().__init__()

        # Encoder
        self.conv1 = medicalnet.conv1
        self.bn1 = medicalnet.bn1
        self.relu = medicalnet.relu
        self.maxpool = medicalnet.maxpool

        self.layer1 = medicalnet.layer1
        self.layer2 = medicalnet.layer2
        self.layer3 = medicalnet.layer3
        self.layer4 = medicalnet.layer4

        # Classification head
        self.pool = nn.AdaptiveAvgPool3d(1)
        # feature_dim = 512 * medicalnet.layer4[-1].expansion
        
        # Infer feature dimension automatically
        last_block = medicalnet.layer4[-1]

        if hasattr(last_block, "bn3"):          # Bottleneck (ResNet/ResNeXt)
            feature_dim = last_block.bn3.num_features
        elif hasattr(last_block, "bn2"):        # BasicBlock
            feature_dim = last_block.bn2.num_features
        else:
            raise ValueError("Unsupported block type")


        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.pool(x)

        return self.classifier(x)


def build_model_medicalnet():
    sets = parse_opts()
    # getting model
    medicalnet, _ = generate_model(sets)
    model = MedicalNetClassifier(medicalnet)

    for p in model.conv1.parameters():
        p.requires_grad = True

    for p in model.bn1.parameters():
        p.requires_grad = True

    for p in model.layer1.parameters():
        p.requires_grad = True

    for p in model.layer2.parameters():
        p.requires_grad = True

    # Train these
    for p in model.layer3.parameters():
        p.requires_grad = True

    for p in model.layer4.parameters():
        p.requires_grad = True

    for p in model.classifier.parameters():
        p.requires_grad = True

    return model

# ===========================================================================
### MedNeXt
# ===========================================================================
class MedNeXtClassifier(nn.Module):
    def __init__(self, mednext):
        super().__init__()

        # Encoder
        self.stem = mednext.stem

        self.enc_block_0 = mednext.enc_block_0
        self.down_0 = mednext.down_0

        self.enc_block_1 = mednext.enc_block_1
        self.down_1 = mednext.down_1

        self.enc_block_2 = mednext.enc_block_2
        self.down_2 = mednext.down_2

        self.enc_block_3 = mednext.enc_block_3
        self.down_3 = mednext.down_3

        # Bottleneck
        self.bottleneck = mednext.bottleneck

        # Infer feature dimension automatically
        # Last block of bottleneck outputs the encoder feature size
        last_block = mednext.bottleneck[-1]

        if hasattr(last_block, "out_channels"):
            feature_dim = last_block.out_channels
        elif hasattr(last_block, "conv3"):
            feature_dim = last_block.conv3.out_channels
        elif hasattr(last_block, "conv2"):
            feature_dim = last_block.conv2.out_channels
        else:
            # Fallback: 16 * base_channels
            feature_dim = 16 * mednext.stem.out_channels

        # Global pooling
        self.pool = nn.AdaptiveAvgPool3d(1)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1)      # Binary classification
        )

    def forward(self, x):
        x = self.stem(x)

        x = self.enc_block_0(x)
        x = self.down_0(x)

        x = self.enc_block_1(x)
        x = self.down_1(x)

        x = self.enc_block_2(x)
        x = self.down_2(x)

        x = self.enc_block_3(x)
        x = self.down_3(x)

        x = self.bottleneck(x)

        x = self.pool(x)

        return self.classifier(x)


def build_model_mednext():
    mednext = create_mednext_v1(
        num_input_channels=1,
        num_classes=2,
        model_id='B',
        kernel_size=3,
        deep_supervision=True
    )
    model = MedNeXtClassifier(mednext)
    return model

# ===========================================================================
### COMMON BUILDER
# ===========================================================================

def print_unique_blocks(module):
    """Prints sub-modules, showing repeated block types only once."""
    seen_types = set()
    
    for name, child in module.named_children():
        child_type = type(child)
        if child_type in seen_types:
            print(f"  ({name}): ... [Repeated instance of {child_type.__name__}]")
        else:
            seen_types.add(child_type)
            print(f"  ({name}): {child}")



def build_model(model_name, VERBOSE=False):
    if 'medicalnet' in model_name:
        model = build_model_medicalnet()

    elif "mednext" in model_name:
        model = build_model_mednext()

    else:
        raise Exception(f"Model {model_name} is not available.")

    if VERBOSE:
        print_unique_blocks(model)
        stats = summary(model, verbose=0)
        total_params = stats.total_params
        print(f"🧠 Total Parameters: {total_params:,}")

    return model
