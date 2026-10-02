# builder.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace

from monai.networks.nets import DenseNet121, EfficientNetBN, DenseNet201
from .medicalnet.model import generate_model
from .mednext.create_mednext_v1 import create_mednext_v1
import ptwt

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


def build_model_medicalnet(depth, backbone, **kwargs):
    sets = get_medicalnet_options("medicalnet", depth, backbone, **kwargs)
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

def get_medicalnet_options(model, medicalnet_depth, backbone_model="resnet", **kwargs):
    """
    Minimal configuration required by MedicalNet generate_model()
    """
    sets = SimpleNamespace(
        model=model,
        medicalnet_depth=medicalnet_depth,
        medicalnet_backbone=backbone_model,
        resnet_shortcut="A" if medicalnet_depth in [18, 34] else "B",

        # Required by MedicalNet
        input_D=128,
        input_H=128,
        input_W=128,

        no_cuda=False,
        gpu_id=[0],

        # Pretraining options
        pretrain_path="NULL",
        resume_path="",

        # Training options (not really used here)
        n_seg_classes=2,
        new_layer_names=["conv_seg"],
        phase='test',

        # Resnext
        n_classes=400,
        sample_size=128, # 128 ? default: 112
        resnext_cardinality=32,
        sample_duration=128, # 128 default: 16
    )
    # Override or add any attributes passed via kwargs
    for key, value in kwargs.items():
        setattr(sets, key, value)

    return sets

# ===========================================================================
### UNET
# ===========================================================================
class UNetClassifier():
    def __init__(self, unet):
        super().__init__()


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


def build_model_mednext(size_mednext='B'):
    mednext = create_mednext_v1(
        num_input_channels=1,
        num_classes=2,
        model_id=size_mednext,
        kernel_size=3,
        deep_supervision=True
    )
    model = MedNeXtClassifier(mednext)
    return model


# ===========================================================================
### SwinTransformer
# ===========================================================================
from monai.networks.nets.swin_unetr import SwinTransformer

class SwinTransformerBinaryClassifier(nn.Module):
    def __init__(
        self,
        in_channels=1,
        feature_size=48,
        depths=(2, 2, 2, 2),
        num_heads=(3, 6, 12, 24),
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.1,
        use_checkpoint=False,
    ):
        super().__init__()

        self.swin = SwinTransformer(
            in_chans=in_channels,
            embed_dim=feature_size,
            window_size=(7, 7, 7),
            patch_size=(2, 2, 2),
            depths=depths,
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=dropout_path_rate,
            norm_layer=nn.LayerNorm,
            use_checkpoint=use_checkpoint,
            spatial_dims=3,
        )

        # Dynamically calculate final feature dimension by running a single forward pass on initialization
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, 64, 64, 64)
            dummy_out = self.swin(dummy_input, normalize=True)
            # MONAI returns a list of stage outputs; pick the deepest feature map
            deepest_feat = dummy_out[-1]
            
            # If channels are at dim 1 (B, C, H, W, D), use shape[1]
            # If channels are at dim -1 (B, H, W, D, C), use shape[-1]
            if deepest_feat.dim() == 5:
                # MONAI Swin outputs (B, C, H, W, D) when normalize=True
                self.hidden_dim = deepest_feat.shape[1] if deepest_feat.shape[1] > deepest_feat.shape[-1] else deepest_feat.shape[-1]
            else:
                self.hidden_dim = feature_size * 16  # fallback (768)

        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, x):
        features = self.swin(x, normalize=True)
        x = features[-1]  # Output shape: (B, 768, H, W, D)

        # Ensure shape is (B, C, H, W, D) if MONAI returns channels last
        if x.dim() == 5 and x.shape[1] != self.hidden_dim:
            x = x.permute(0, 4, 1, 2, 3).contiguous()

        # Pool spatial dimensions (H, W, D) -> (1, 1, 1)
        x = self.pool(x)

        # Flatten to (B, 768)
        x = torch.flatten(x, 1)

        # Final prediction (B, 1)
        x = self.head(x)

        return x

# ===========================================================================
### ATTENTION DENSENET
# ===========================================================================
class SelfAttention3D(nn.Module):
    """
    Non-Local Block 3D avec initialisation à zéro (Zero-Init).
    Permet au réseau de commencer comme un DenseNet classique puis d'apprendre l'attention.
    """
    def __init__(self, in_channels, reduction=8):
        super().__init__()
        self.in_channels = in_channels
        self.inter_channels = max(in_channels // reduction, 1)
        
        self.g = nn.Conv3d(in_channels, self.inter_channels, kernel_size=1)
        self.theta = nn.Conv3d(in_channels, self.inter_channels, kernel_size=1)
        self.phi = nn.Conv3d(in_channels, self.inter_channels, kernel_size=1)
        
        self.W = nn.Conv3d(self.inter_channels, in_channels, kernel_size=1)
        
        # Initialisation à zéro
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        b, c, d, h, w = x.size()
        
        g_x = self.g(x).view(b, self.inter_channels, -1) 
        g_x = g_x.permute(0, 2, 1) 
        
        theta_x = self.theta(x).view(b, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1) 
        phi_x = self.phi(x).view(b, self.inter_channels, -1) 
        
        f = torch.matmul(theta_x, phi_x) 
        f_div_C = F.softmax(f, dim=-1)
        
        y = torch.matmul(f_div_C, g_x) 
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(b, self.inter_channels, d, h, w) 
        
        W_y = self.W(y)
        return x + self.gamma * W_y

class AttentionDenseNet(nn.Module):
    """
    Wrapper générique qui injecte l'attention dans un DenseNet MONAI (121, 201, etc.).
    """
    def __init__(self, base_model):
        super().__init__()
        self.features = base_model.features
        self.class_layers = base_model.class_layers
        
        # Déduction dynamique du nombre de canaux après transition2 et transition3
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 64, 64, 64) 
            x = dummy
            for i in range(8): 
                x = self.features[i](x)
            ch_trans2 = x.shape[1]
            
            for i in range(8, 10): 
                x = self.features[i](x)
            ch_trans3 = x.shape[1]
            
        self.attn2 = SelfAttention3D(ch_trans2)
        self.attn3 = SelfAttention3D(ch_trans3)
        
    def forward(self, x):
        for i in range(8):
            x = self.features[i](x)
        x = self.attn2(x) 
        
        for i in range(8, 10):
            x = self.features[i](x)
        x = self.attn3(x) 
        
        for i in range(10, len(self.features)):
            x = self.features[i](x)
            
        x = self.class_layers(x)
        return x

# ===========================================================================
### Simple Siamese (MIP Axial)
# ===========================================================================
class SimpleSiameseCNN(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 128 -> 64
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 64 -> 32
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 32 -> 16
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)  # Sortie : (B, 256, 1, 1)
        )
        
        feat_dim = 256
        
        self.head = nn.Sequential(
            nn.Linear(feat_dim * 4, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1)
        )

    def forward(self, x_L, x_R):
        feat_L = self.backbone(x_L).view(x_L.size(0), -1)  # (B, 256)
        feat_R = self.backbone(x_R).view(x_R.size(0), -1)  # (B, 256)
        
        diff = torch.abs(feat_L - feat_R)
        prod = feat_L * feat_R
        
        fusion = torch.cat([feat_L, feat_R, diff, prod], dim=1)  # (B, 1024)
        return self.head(fusion)

def build_model_simplesiamese():
    return SimpleSiameseCNN(dropout_rate=0.3)

# ===========================================================================
### Multi-View Dual-Stream Siamese (2.5D - Axiale + Coronale)
# ===========================================================================
def make_siamese_backbone(in_channels, feat_dim=128):
    """Encodeur 2D partagé (poids initialisés aléatoirement)."""
    return nn.Sequential(
        nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),  # 128 -> 64
        
        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),  # 64 -> 32
        
        nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),  # 32 -> 16
        
        nn.Conv2d(128, feat_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(feat_dim),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1)  # Sortie : (B, feat_dim, 1, 1)
    )

class MultiViewSiameseCNN(nn.Module):
    def __init__(self, in_channels_ax=6, in_channels_cor=6, feat_dim=128, dropout_rate=0.3):
        super().__init__()
        # Encodeurs dédiés pour chaque vue
        self.backbone_ax = make_siamese_backbone(in_channels_ax, feat_dim)
        self.backbone_cor = make_siamese_backbone(in_channels_cor, feat_dim)
            
        # Fusion : (Axial L, R, |L-R|, L*R) + (Coronal L, R, |L-R|, L*R)
        total_fusion_dim = (feat_dim * 4) + (feat_dim * 4)
        
        self.head = nn.Sequential(
            nn.Linear(total_fusion_dim, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1)
        )

    def _extract_bilateral(self, backbone, x_L, x_R):
        feat_L = backbone(x_L).view(x_L.size(0), -1)
        feat_R = backbone(x_R).view(x_R.size(0), -1)
        diff = torch.abs(feat_L - feat_R)
        prod = feat_L * feat_R
        return torch.cat([feat_L, feat_R, diff, prod], dim=1)

    def forward(self, ax_pair, cor_pair):
        ax_L, ax_R = ax_pair
        cor_L, cor_R = cor_pair
        
        fusion_ax = self._extract_bilateral(self.backbone_ax, ax_L, ax_R)
        fusion_cor = self._extract_bilateral(self.backbone_cor, cor_L, cor_R)
            
        total_fusion = torch.cat([fusion_ax, fusion_cor], dim=1)
        return self.head(total_fusion)

def build_model_multiview_siamese():
    return MultiViewSiameseCNN(in_channels_ax=6, in_channels_cor=6, dropout_rate=0.3)

# ===========================================================================
# BACKBONE Siamese mashup COMMUN
# ===========================================================================
def create_mashup_backbone(in_channels=3):
    return nn.Sequential(
        nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2), 
        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2), 
        nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2), 
        nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten()
    )

def _siamese_fusion(feat_A, feat_B, projector):
    diff = torch.abs(feat_A - feat_B)
    prod = feat_A * feat_B
    return projector(torch.cat([feat_A, feat_B, diff, prod], dim=1))

# ===========================================================================
### 1. Siamese Mashup Attention CLASSIQUE (3 Branches : Global, Fade, Tight)
# ===========================================================================
class MashupAttentionCNN(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super().__init__()
        self.enc_global = create_mashup_backbone()
        self.enc_fade   = create_mashup_backbone()
        self.enc_tight  = create_mashup_backbone()
        
        self.proj_global = nn.Linear(256 * 4, 256)
        self.proj_fade   = nn.Linear(256 * 4, 256)
        self.proj_tight  = nn.Linear(256 * 4, 256)
        
        self.attention = nn.Sequential(nn.Linear(256, 128), nn.Tanh(), nn.Linear(128, 1))
        
        self.head = nn.Sequential(
            nn.LayerNorm(256), nn.SiLU(), nn.Dropout(dropout_rate), nn.Linear(256, 1)
        )

    def forward(self, img_G, img_G_flip, img_L_fade, img_R_fade, img_L_tight, img_R_tight):
        vec_global = _siamese_fusion(self.enc_global(img_G), self.enc_global(img_G_flip), self.proj_global)
        vec_fade   = _siamese_fusion(self.enc_fade(img_L_fade), self.enc_fade(img_R_fade), self.proj_fade)
        vec_tight  = _siamese_fusion(self.enc_tight(img_L_tight), self.enc_tight(img_R_tight), self.proj_tight)
        
        stacked_views = torch.stack([vec_global, vec_fade, vec_tight], dim=1)
        attn_weights = torch.softmax(self.attention(stacked_views), dim=1)
        final_vec = torch.sum(stacked_views * attn_weights, dim=1)
        
        return self.head(final_vec)

def build_model_siamese_mashup():
    return MashupAttentionCNN(dropout_rate=0.3)


# ===========================================================================
### 2. Siamese Mashup Attention ALIGN (4 Branches : + Align Tight)
# ===========================================================================
class MashupAttentionAlignCNN(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super().__init__()
        self.enc_global = create_mashup_backbone()
        self.enc_fade   = create_mashup_backbone()
        self.enc_tight  = create_mashup_backbone()
        self.enc_align  = create_mashup_backbone() 
        
        self.proj_global = nn.Linear(256 * 4, 256)
        self.proj_fade   = nn.Linear(256 * 4, 256)
        self.proj_tight  = nn.Linear(256 * 4, 256)
        self.proj_align  = nn.Linear(256 * 4, 256) 
        
        self.attention = nn.Sequential(nn.Linear(256, 128), nn.Tanh(), nn.Linear(128, 1))
        
        self.head = nn.Sequential(
            nn.LayerNorm(256), nn.SiLU(), nn.Dropout(dropout_rate), nn.Linear(256, 1)
        )

    def forward(self, img_G, img_G_flip, img_L_fade, img_R_fade, img_L_tight, img_R_tight, img_L_align, img_R_align):
        vec_global = _siamese_fusion(self.enc_global(img_G), self.enc_global(img_G_flip), self.proj_global)
        vec_fade   = _siamese_fusion(self.enc_fade(img_L_fade), self.enc_fade(img_R_fade), self.proj_fade)
        vec_tight  = _siamese_fusion(self.enc_tight(img_L_tight), self.enc_tight(img_R_tight), self.proj_tight)
        vec_align  = _siamese_fusion(self.enc_align(img_L_align), self.enc_align(img_R_align), self.proj_align)
        
        stacked_views = torch.stack([vec_global, vec_fade, vec_tight, vec_align], dim=1)
        attn_weights = torch.softmax(self.attention(stacked_views), dim=1)
        final_vec = torch.sum(stacked_views * attn_weights, dim=1)
        
        return self.head(final_vec)

def build_model_siamese_mashup_align():
    return MashupAttentionAlignCNN(dropout_rate=0.3)


# ===========================================================================
### 2. Siamese Mashup Attention WAVELET (4 Branches : Global, Fade, Tight, CWT)
# ===========================================================================
def compute_dwt_2d_tensor(x, wave='db1'):
    """
    Applique la DWT 2D sur un batch d'images (B, C, H, W).
    Concatène les sous-bandes (LL, LH, HL, HH) le long du canal -> (B, C*4, H//2, W//2).
    """
    LL, (LH, HL, HH) = ptwt.wavedec2(x, wavelet=wave, level=1)
    return torch.cat([LL, LH, HL, HH], dim=1) # (B, C*4, H/2, W/2)


class MashupAttentionCNNLinear(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super().__init__()
        self.enc_global = create_mashup_backbone(in_channels=3)
        self.enc_fade   = create_mashup_backbone(in_channels=3)
        self.enc_tight  = create_mashup_backbone(in_channels=3)
        self.enc_dwt    = create_mashup_backbone(in_channels=12) # 4 sub-bands * 3 RGB channels = 12 channels
        
        self.proj_global = nn.Linear(256 * 4, 256)
        self.proj_fade   = nn.Linear(256 * 4, 256)
        self.proj_tight  = nn.Linear(256 * 4, 256)
        self.proj_dwt    = nn.Linear(256 * 4, 256)
        
        # Attention agrège désormais 4 vues : [Global, Fade, Tight, DWT]
        self.attention = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
        self.head = nn.Sequential(
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 1)
        )

    def _siamese_fusion(self, feat_A, feat_B, projector):
        diff = torch.abs(feat_A - feat_B)
        prod = feat_A * feat_B
        return projector(torch.cat([feat_A, feat_B, diff, prod], dim=1))

    def forward(self, img_G, img_G_flip, img_L_fade, img_R_fade, img_L_tight, img_R_tight, img_L, img_R):
        # 1. Dynamic 2D DWT calculation on GPU
        dwt_L = compute_dwt_2d_tensor(img_L, wave='db1')
        dwt_R = compute_dwt_2d_tensor(img_R, wave='db1')

        # 2. Forward through the 4 siamese encoders
        vec_global = self._siamese_fusion(self.enc_global(img_G), self.enc_global(img_G_flip), self.proj_global)
        vec_fade   = self._siamese_fusion(self.enc_fade(img_L_fade), self.enc_fade(img_R_fade), self.proj_fade)
        vec_tight  = self._siamese_fusion(self.enc_tight(img_L_tight), self.enc_tight(img_R_tight), self.proj_tight)
        vec_dwt    = self._siamese_fusion(self.enc_dwt(dwt_L), self.enc_dwt(dwt_R), self.proj_dwt)
        
        # 3. Empirical Multi-View Attention Fusion across 4 streams
        stacked_views = torch.stack([vec_global, vec_fade, vec_tight, vec_dwt], dim=1) # (B, 4, 256)
        attn_weights = torch.softmax(self.attention(stacked_views), dim=1)              # (B, 4, 1)
        final_vec = torch.sum(stacked_views * attn_weights, dim=1)                      # (B, 256)
        
        return self.head(final_vec)

def build_model_siamese_mashup_wavelet_linear():
    return MashupAttentionCNNLinear(dropout_rate=0.3)    


class MashupAttentionCNNWaveletAttn(nn.Module):
    def __init__(self, dropout_rate=0.3, in_channels=3, embed_dim=256):
        super().__init__()
        
        # 1. Backbones for the 4 views
        self.enc_global = create_mashup_backbone(in_channels=in_channels)
        self.enc_fade   = create_mashup_backbone(in_channels=in_channels)
        self.enc_tight  = create_mashup_backbone(in_channels=in_channels)
        self.enc_dwt    = create_mashup_backbone(in_channels=4 * in_channels) # 4 sub-bands * in_channels
        
        # 2. Projectors (mapping Siamese features to embed_dim)
        self.proj_global = nn.Linear(256 * 4, embed_dim)
        self.proj_fade   = nn.Linear(256 * 4, embed_dim)
        self.proj_tight  = nn.Linear(256 * 4, embed_dim)
        self.proj_dwt    = nn.Linear(256 * 4, embed_dim)
        
        # 3. Learnable [CLS] Token for cross-view aggregation
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        
        # 4. Custom 2-Layer Transformer Encoder with Attention Weight Extraction
        # Layer 1: Standard Transformer Encoder Layer
        self.tf_layer1 = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=4,
            dim_feedforward=512,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        
        # Layer 2: Explicit MultiheadAttention + FeedForward to extract attention weights
        self.attn2 = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, embed_dim),
            nn.Dropout(0.1)
        )
        
        # 5. Final Classification Head
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embed_dim, 1)
        )

    def _siamese_fusion(self, feat_A, feat_B, projector):
        diff = torch.abs(feat_A - feat_B)
        prod = feat_A * feat_B
        return projector(torch.cat([feat_A, feat_B, diff, prod], dim=1))

    def forward(self, img_G, img_G_flip, img_L_fade, img_R_fade, img_L_tight, img_R_tight, img_L, img_R):
        # 1. Calcul dynamique des sous-bandes DWT sur GPU (16x16x12)
        dwt_L = compute_dwt_2d_tensor(img_L, wave='db1')
        dwt_R = compute_dwt_2d_tensor(img_R, wave='db1')

        # 2. Forward à travers les 4 encoders siamois -> (B, 256) each
        vec_global = self._siamese_fusion(self.enc_global(img_G), self.enc_global(img_G_flip), self.proj_global)
        vec_fade   = self._siamese_fusion(self.enc_fade(img_L_fade), self.enc_fade(img_R_fade), self.proj_fade)
        vec_tight  = self._siamese_fusion(self.enc_tight(img_L_tight), self.enc_tight(img_R_tight), self.proj_tight)
        vec_dwt    = self._siamese_fusion(self.enc_dwt(dwt_L), self.enc_dwt(dwt_R), self.proj_dwt)
        
        # 3. Stack views into sequence: (B, 4, 256)
        stacked_views = torch.stack([vec_global, vec_fade, vec_tight, vec_dwt], dim=1)
        
        # 4. Prepend [CLS] token -> (B, 5, 256) where index 0 is [CLS]
        batch_size = stacked_views.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, stacked_views], dim=1)
        
        # 5. Transformer Layer 1
        x = self.tf_layer1(x)
        
        # 6. Transformer Layer 2 with Attention Weights Extraction
        # Extract attention matrix from [CLS] token (query) to all tokens (keys)
        attn_out, raw_attn_weights = self.attn2(x, x, x, average_attn_weights=True)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        
        # 7. Extract fused CLS feature vector (B, 256)
        fused_cls_vec = x[:, 0, :]
        
        # 8. Extract attention weights assigned by [CLS] (index 0) to the 4 views (indices 1..4)
        # Reshape to (B, 4, 1) to perfectly match your training loop's expected format
        cls_attn_to_views = raw_attn_weights[:, 0, 1:].unsqueeze(-1) # (B, 4, 1)
        
        # Return tuple matching signature: logits
        return self.head(fused_cls_vec)


def build_model_siamese_mashup_wavelet_attn():
    return MashupAttentionCNNWaveletAttn(dropout_rate=0.3)


# ===========================================================================
### DEUX PHASES - MULTI-BRANCH (Mean Aggregator)
# ===========================================================================
import ptwt

def create_twophase_backbone(in_channels=3, w=32):
    """Backbone CNN partagé par les branches images."""
    return nn.Sequential(
        nn.Conv2d(in_channels, w, 3, 1, 1), nn.BatchNorm2d(w), nn.ReLU(), nn.MaxPool2d(2, 2),
        nn.Conv2d(w, 2 * w, 3, 1, 1), nn.BatchNorm2d(2 * w), nn.ReLU(), nn.MaxPool2d(2, 2),
        nn.Conv2d(2 * w, 4 * w, 3, 1, 1), nn.BatchNorm2d(4 * w), nn.ReLU(), nn.MaxPool2d(2, 2),
        nn.Conv2d(4 * w, 8 * w, 3, 1, 1), nn.BatchNorm2d(8 * w), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten()
    )

class SpineEncoder(nn.Module):
    """Encodeur BiLSTM + Attention pour la branche spine."""
    def __init__(self, n_feat=3, hidden=64, layers=2, dropout=0.3, n_heads=4, out_dim=256):
        super().__init__()
        d = 2 * hidden
        self.lstm = nn.LSTM(n_feat, hidden, layers, batch_first=True, bidirectional=True, dropout=dropout if layers>1 else 0.)
        self.self_attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.norm_s = nn.LayerNorm(d)
        self.cross_attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.norm_c = nn.LayerNorm(d)
        self.pool = nn.Sequential(nn.Linear(d, 64), nn.Tanh(), nn.Linear(64, 1))
        self.proj = nn.Linear(4 * d, out_dim)

    def _pool(self, h):
        w = torch.softmax(self.pool(h), 1)
        return (h * w).sum(1), w.squeeze(-1)

    def forward(self, xL, xR):
        hL, _ = self.lstm(xL)
        hR, _ = self.lstm(xR)
        sL, _ = self.self_attn(hL, hL, hL)
        sR, _ = self.self_attn(hR, hR, hR)
        hL, hR = self.norm_s(hL + sL), self.norm_s(hR + sR)
        cL, _ = self.cross_attn(hL, hR, hR)
        cR, _ = self.cross_attn(hR, hL, hL)
        hL, hR = self.norm_c(hL + cL), self.norm_c(hR + cR)
        a, _ = self._pool(hL)
        b, _ = self._pool(hR)
        return self.proj(torch.cat([a, b, (a - b).abs(), a * b], 1))

class MeanAgg(nn.Module):
    """Agrégateur par moyenne (Config: AGG_TYPE = 'mean')."""
    def __init__(self, n, d, dr=0.3): 
        super().__init__()
        self.norm = nn.LayerNorm(d)
        
    def forward(self, v, qual_bias=None):
        B, N, _ = v.shape
        # Retourne (feature_fusionnée, poids_fictifs_uniformes)
        return self.norm(v.mean(1)), torch.full((B, N), 1. / N, device=v.device)

class TwoPhaseMultiBranchModel(nn.Module):
    """Modèle de production restreint à la configuration choisie (6 branches, MeanAgg)."""
    def __init__(self, n_scalars, embed_dim=256):
        super().__init__()
        self.branches = ('global', 'fade', 'tight', 'dwt', 'scalars', 'spine')
        self.enc = nn.ModuleDict()
        self.proj = nn.ModuleDict()
        self.aux = nn.ModuleDict()
        
        # Dimensions issues de votre config avec USE_MASK=True pour fade/tight
        dims = {'global_ch': 3, 'fade_ch': 5, 'tight_ch': 5, 'dwt_ch': 12, 'n_scalars': n_scalars}
        
        for b in self.branches:
            if b in ('global', 'fade', 'tight', 'dwt'):
                self.enc[b] = create_twophase_backbone(dims[f'{b}_ch'], 32)
                self.proj[b] = nn.Linear(8 * 32 * 4, embed_dim)
            elif b == 'scalars':
                self.enc[b] = nn.Sequential(
                    nn.Linear(dims['n_scalars'], 128), nn.LayerNorm(128), nn.ReLU(), 
                    nn.Dropout(0.20), nn.Linear(128, embed_dim)
                )
            elif b == 'spine':
                self.enc[b] = SpineEncoder(out_dim=embed_dim)
                
            # Les têtes aux doivent être présentes pour charger les poids, même si ignorées en inférence
            self.aux[b] = nn.Sequential(nn.LayerNorm(embed_dim), nn.SiLU(), nn.Dropout(0.3), nn.Linear(embed_dim, 1))

        self.agg = MeanAgg(len(self.branches), embed_dim)
        self.head = nn.Sequential(nn.LayerNorm(embed_dim), nn.SiLU(), nn.Dropout(0.3), nn.Linear(embed_dim, 1))
        
        # Compatibilité d'attributs avec votre logique de forward
        self.qual = None 

    def _siam(self, b, A, B_):
        fa, fb = self.enc[b](A), self.enc[b](B_)
        return self.proj[b](torch.cat([fa, fb, (fa - fb).abs(), fa * fb], 1))

    def encode(self, b, batch):
        if b == 'global':  return self._siam(b, batch['G'], batch['G_flip'])
        if b == 'fade':    return self._siam(b, batch['L_fade'], batch['R_fade'])
        if b == 'tight':   return self._siam(b, batch['L_tight'], batch['R_tight'])
        if b == 'dwt':     return self._siam(b, batch['L_dwt'], batch['R_dwt'])
        if b == 'scalars': return self.enc[b](batch['scalars'])
        if b == 'spine':   return self.enc[b](batch['spine_L'], batch['spine_R'])

    def forward(self, batch):
        vecs = [self.encode(b, batch) for b in self.branches]
        stack = torch.stack(vecs, 1)
        h, _ = self.agg(stack)
        return self.head(h)

def build_model_twophase_mean(n_scalars):
    """Fonction appelée par le builder commun."""
    return TwoPhaseMultiBranchModel(n_scalars=n_scalars)

# ===========================================================================
# BACKBONE SIAMESE INTERNAL ENSEMBLE (LATE FUSION)
# ===========================================================================
import ptwt

# 🔑 Fonction utilitaire nécessaire pour le forward (comme dans le train)
def compute_dwt_LL_tensor(x, wave='db1'):
    LL, _ = ptwt.wavedec2(x, wavelet=wave, level=1)
    return LL

def create_internal_backbone():
    return nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2), 
        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2), 
        nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2), 
        nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten()
    )

class MashupInternalEnsembleCNN(nn.Module):
    # 🔑 Ajout de use_dwt
    def __init__(self, dropout_rate=0.3, use_dwt=False):
        super().__init__()
        self.use_dwt = use_dwt
        self.enc_global, self.enc_fade, self.enc_tight = create_internal_backbone(), create_internal_backbone(), create_internal_backbone()
        self.proj_global, self.proj_fade, self.proj_tight = nn.Linear(1024, 256), nn.Linear(1024, 256), nn.Linear(1024, 256)
        self.head_global = nn.Sequential(nn.LayerNorm(256), nn.SiLU(), nn.Dropout(dropout_rate), nn.Linear(256, 1))
        self.head_fade   = nn.Sequential(nn.LayerNorm(256), nn.SiLU(), nn.Dropout(dropout_rate), nn.Linear(256, 1))
        self.head_tight  = nn.Sequential(nn.LayerNorm(256), nn.SiLU(), nn.Dropout(dropout_rate), nn.Linear(256, 1))

        # 🔑 Instanciation des couches conditionnelles DWT
        if self.use_dwt:
            self.enc_dwt = create_internal_backbone()
            self.proj_dwt = nn.Linear(1024, 256)
            self.head_dwt = nn.Sequential(nn.LayerNorm(256), nn.SiLU(), nn.Dropout(dropout_rate), nn.Linear(256, 1))

    def _siamese_fusion(self, feat_A, feat_B, projector):
        return projector(torch.cat([feat_A, feat_B, torch.abs(feat_A - feat_B), feat_A * feat_B], dim=1))

    # 🔑 Signature ajustée avec des None pour rétrocompatibilité
    def forward(self, img_G, img_G_flip, img_L_fade, img_R_fade, img_L_tight, img_R_tight, img_L=None, img_R=None):
        v_glob  = self._siamese_fusion(self.enc_global(img_G), self.enc_global(img_G_flip), self.proj_global)
        v_fade  = self._siamese_fusion(self.enc_fade(img_L_fade), self.enc_fade(img_R_fade), self.proj_fade)
        v_tight = self._siamese_fusion(self.enc_tight(img_L_tight), self.enc_tight(img_R_tight), self.proj_tight)
        
        logit_glob, logit_fade, logit_tight = self.head_global(v_glob), self.head_fade(v_fade), self.head_tight(v_tight)
        
        # 🔑 Forward conditionnel
        if self.use_dwt and img_L is not None and img_R is not None:
            dwt_L = compute_dwt_LL_tensor(img_L)
            dwt_R = compute_dwt_LL_tensor(img_R)
            v_dwt = self._siamese_fusion(self.enc_dwt(dwt_L), self.enc_dwt(dwt_R), self.proj_dwt)
            logit_dwt = self.head_dwt(v_dwt)
            
            mean_logit = (logit_glob + logit_fade + logit_tight + logit_dwt) / 4.0
            return mean_logit, logit_glob, logit_fade, logit_tight, logit_dwt
        else:
            mean_logit = (logit_glob + logit_fade + logit_tight) / 3.0
            return mean_logit, logit_glob, logit_fade, logit_tight

def build_model_siamese_internal(use_dwt=False):
    return MashupInternalEnsembleCNN(dropout_rate=0.3, use_dwt=use_dwt)

# ===========================================================================
### Mini-CNN 3D Siamois (Cubes)
# ===========================================================================
class ResBlock3D(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_c, out_c, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_c)
        self.conv2 = nn.Conv3d(out_c, out_c, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_c)
        self.act = nn.SiLU(inplace=True)
        self.downsample = nn.Sequential(
            nn.Conv3d(in_c, out_c, 1, stride=stride, bias=False),
            nn.BatchNorm3d(out_c)
        ) if stride != 1 or in_c != out_c else nn.Identity()

    def forward(self, x):
        res = self.downsample(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + res)

class MiniSiamese3D(nn.Module):
    def __init__(self, feat_dim=128, dropout_rate=0.3):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.SiLU(inplace=True),
            ResBlock3D(32, 64, stride=2),   
            ResBlock3D(64, 128, stride=2),  
            ResBlock3D(128, feat_dim, stride=2), 
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten()
        )
        self.head = nn.Sequential(
            nn.Linear(feat_dim * 4, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1)
        )

    def forward(self, x_L, x_R):
        feat_L = self.backbone(x_L) 
        feat_R = self.backbone(x_R) 
        diff = torch.abs(feat_L - feat_R)
        prod = feat_L * feat_R
        fusion = torch.cat([feat_L, feat_R, diff, prod], dim=1) 
        return self.head(fusion)

def build_model_minicnn3d():
    return MiniSiamese3D(feat_dim=128, dropout_rate=0.3)

# ===========================================================================
### COMMON BUILDER
# ===========================================================================
def build_model(model_name, **kwargs):
    model_name = model_name.lower()
    
    if 'medicalnet' in model_name:
        model = build_model_medicalnet() # Géré différemment via _instantiate_model de inference_3d.py

    elif "mednext" in model_name:
        # On isole ce qui vient après "mednext" (ex: "_b", "b", "-b")
        suffix = model_name.split('mednext')[-1]
        
        # On déduit la taille de manière robuste
        mednext_size = 'B'  # Taille par défaut
        if 's' in suffix: mednext_size = 'S'
        elif 'm' in suffix: mednext_size = 'M'
        elif 'l' in suffix: mednext_size = 'L'
        elif 'b' in suffix: mednext_size = 'B'
        
        model = build_model_mednext(mednext_size)

    elif "swintf" in model_name:
        model = SwinTransformerBinaryClassifier(
            in_channels=1,
            feature_size=48,
        )
    
    elif 'densenet' in model_name:
        # 1. Choix du backbone (121 ou 201)
        if '201' in model_name:
            base_model = DenseNet201(spatial_dims=3, in_channels=1, out_channels=1, dropout_prob=0.2)
        else: # Par défaut 121
            base_model = DenseNet121(spatial_dims=3, in_channels=1, out_channels=1, dropout_prob=0.2)
            
        # 2. Ajout de l'attention si demandé
        if 'attention' in model_name or 'attn' in model_name:
            model = AttentionDenseNet(base_model)
        else:
            model = base_model
    
    elif 'efficientnet' in model_name:
        model = EfficientNetBN(
            model_name=model_name,
            spatial_dims=3,
            in_channels=1,
            num_classes=1,
        )

    elif model_name == 'minicnn3d_siamese':
        model = build_model_minicnn3d()

    elif 'simple_siamese' in model_name:
        model = build_model_simplesiamese()

    elif 'multiview_siamese' in model_name:
        model = build_model_multiview_siamese()

    # 🔑 Branche pour la version 3 Vues
    elif model_name == 'siamese_mashup':
        model = build_model_siamese_mashup()

    # 🔑 Branche pour la version 4 Vues
    elif model_name == 'siamese_mashup_align':
        model = build_model_siamese_mashup_align()
    
    # 🔑 Branche pour la version 4 Vues Wavelet
    elif model_name == 'siamese_mashup_wavelet_attn':
        model = build_model_siamese_mashup_wavelet_attn()

    # 🔑 Branche pour la version 4 Vues Wavelet
    elif model_name == 'siamese_mashup_wavelet_linear':
        model = build_model_siamese_mashup_wavelet_linear()

    elif model_name == 'twophase_mean':
        n_scalars = kwargs.get('n_scalars', 0) # Assurez-vous de passer le nombre de features scalaires
        model = build_model_twophase_mean(n_scalars=n_scalars)

    elif model_name == 'siamese_internal':
        # 🔑 On intercepte l'argument et on le passe
        use_dwt = kwargs.get('use_dwt', False)
        model = build_model_siamese_internal(use_dwt=use_dwt)

    return model
