"""
inference_2d_sagcorax_rnn.py
============================
Classe d'inférence orientée objet pour les modèles tri-view (Axial + Coronal + Sagittal) 2D CNN + RNN.
Encapsule le chargement des K-folds et la prédiction pour les modèles Tri-Branch 2D-RNN.
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.v2 as transforms
import timm

# -----------------------------------------------------------------------------
# 1. Architecture du Modèle Tri-Branch (Axial + Coronal + Sagittal)
# -----------------------------------------------------------------------------
class TemporalAttention(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        weights = torch.softmax(self.attn(x), dim=1)  # (B, S, 1)
        context = torch.sum(x * weights, dim=1)       # (B, feature_dim)
        return context


class ViewBranch(nn.Module):
    """Gère une branche de vue individuelle (Axial, Coronal ou Sagittal)."""
    def __init__(self, backbone_name="resnet18d", pretrained=False, rnn_type="lstm", rnn_hidden=128, num_layers=2):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            in_chans=3,
            num_classes=0
        )
        feature_dim = self.backbone.num_features

        rnn_type = rnn_type.lower()
        rnn_cls = nn.LSTM if rnn_type == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size=feature_dim,
            hidden_size=rnn_hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2 if num_layers > 1 else 0.0
        )
        
        self.rnn_out_dim = rnn_hidden * 2
        self.attention = TemporalAttention(feature_dim=self.rnn_out_dim)

    def forward(self, x):
        B, S, C, H, W = x.shape
        x_flat = x.view(B * S, C, H, W)
        features_2d = self.backbone(x_flat)
        
        sequence_input = features_2d.view(B, S, -1)
        rnn_out, _ = self.rnn(sequence_input)
        context_vector = self.attention(rnn_out)  # (B, rnn_out_dim)
        return context_vector


class TriBranch_CNN2D_RNN_Model(nn.Module):
    """
    Modèle Tri-Branch identique à l'entraînement:
    Combine trois ViewBranch (Axial, Coronal, Sagittal) suivies d'une concaténation
    et d'un classificateur dense à 3 couches (256 -> 64 -> 1).
    """
    def __init__(self, backbone_name="resnet18d", pretrained=False, rnn_type="lstm", rnn_hidden=128, num_layers=2):
        super().__init__()
        
        # Tri Feature Extractors
        self.axial_branch = ViewBranch(backbone_name, pretrained, rnn_type, rnn_hidden, num_layers)
        self.coronal_branch = ViewBranch(backbone_name, pretrained, rnn_type, rnn_hidden, num_layers)
        self.sagittal_branch = ViewBranch(backbone_name, pretrained, rnn_type, rnn_hidden, num_layers)
        
        fused_dim = (
            self.axial_branch.rnn_out_dim + 
            self.coronal_branch.rnn_out_dim + 
            self.sagittal_branch.rnn_out_dim
        )

        # Combined Classifier
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x_axial, x_coronal, x_sagittal):
        axial_feat = self.axial_branch(x_axial)          # (B, rnn_out_dim)
        coronal_feat = self.coronal_branch(x_coronal)    # (B, rnn_out_dim)
        sagittal_feat = self.sagittal_branch(x_sagittal) # (B, rnn_out_dim)
        
        fused_feat = torch.cat([axial_feat, coronal_feat, sagittal_feat], dim=-1)  # (B, rnn_out_dim * 3)
        logits = self.classifier(fused_feat)
        return logits


# -----------------------------------------------------------------------------
# 2. Classe de Base pour l'Ensemble Inférence
# -----------------------------------------------------------------------------
class BaseEnsembleSagCorAx2DRNN:
    """
    Classe d'inférence factorisée pour charger un ensemble de modèles Tri-Branch
    (Axial + Coronal + Sagittal) et moyenner leurs prédictions.
    """

    def __init__(
        self,
        backbone_name="resnet18d",
        rnn_type="lstm",
        rnn_hidden=128,
        models_dir="",
        prefix=None,
        num_folds=5,
        img_size=(128, 128),
        device=None,
    ):
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.models = []
        self.backbone_name = backbone_name
        self.rnn_type = rnn_type.lower()
        self.rnn_hidden = rnn_hidden

        self.prefix = prefix if prefix else f"2dsagcorax_{self.backbone_name}_{self.rnn_type}"

        # Transformations V2
        self.transform = transforms.Compose([
            transforms.Resize(img_size, antialias=True),
        ])

        print(
            f"⚙️ Chargement de l'ensemble Tri-View 2D-RNN ({self.backbone_name} + {self.rnn_type.upper()}) "
            f"[Préfixe: '{self.prefix}', {num_folds} folds] sur {self.device}"
        )

        for fold in range(num_folds):
            model_path = os.path.join(models_dir, f"{self.prefix}_fold_{fold}.pth")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Poids introuvables : {model_path}")

            model = self._instantiate_model()
            checkpoint = torch.load(model_path, map_location=self.device)

            if isinstance(checkpoint, dict):
                if "model_state_dict" in checkpoint:
                    state_dict = checkpoint["model_state_dict"]
                elif "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint

            model.load_state_dict(state_dict, strict=True)
            model.to(self.device)
            model.eval()

            self.models.append(model)

    def _instantiate_model(self):
        return TriBranch_CNN2D_RNN_Model(
            backbone_name=self.backbone_name,
            pretrained=False,
            rnn_type=self.rnn_type,
            rnn_hidden=self.rnn_hidden,
        )

    def _preprocess_slice_sequence(self, slices_seq, target_num_slices=20, index_key="z_index"):
        """
        Convertit une séquence de coupes (liste de dicts/numpy ou Tensor) 
        en un Tensor PyTorch de forme (1, S, 3, H, W).
        """
        if isinstance(slices_seq, list):
            if not slices_seq:
                return None

            # 1. Tri par index si dictionnaire
            if isinstance(slices_seq[0], dict) and index_key in slices_seq[0]:
                slices_seq = sorted(slices_seq, key=lambda x: x[index_key])

            # 2. Ré-échantillonnage
            num_slices = len(slices_seq)
            if num_slices != target_num_slices:
                indices = np.linspace(0, num_slices - 1, target_num_slices).round().astype(int)
                slices_seq = [slices_seq[i] for i in indices]

            slice_tensors = []
            for slice_item in slices_seq:
                img_gray = slice_item["image"] if isinstance(slice_item, dict) else slice_item

                if img_gray.ndim == 3 and img_gray.shape[-1] in [1, 3]:
                    img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)

                img_float = img_gray.astype(np.float32) / 255.0
                img_3ch = np.stack([img_float] * 3, axis=0)

                tensor = torch.from_numpy(img_3ch)
                if self.transform:
                    tensor = self.transform(tensor)

                slice_tensors.append(tensor)

            sequence_tensor = torch.stack(slice_tensors, dim=0)  # (S, 3, H, W)
            return sequence_tensor.unsqueeze(0).to(self.device)   # (1, S, 3, H, W)

        elif isinstance(slices_seq, torch.Tensor):
            if slices_seq.dim() == 4:
                return slices_seq.unsqueeze(0).to(self.device)
            elif slices_seq.dim() == 5:
                return slices_seq.to(self.device)
            else:
                raise ValueError(f"Shape inattendue pour le Tensor d'entrée : {slices_seq.shape}")
        else:
            raise TypeError(f"Type d'entrée non supporté : {type(slices_seq)}")

    def predict(self, slices_input, slices_coronal=None, slices_sagittal=None, target_num_slices=20):
        """
        Inférence sur les trois vues (Axial + Coronal + Sagittal).

        :param slices_input: Soit le dictionnaire tri-view 
                             ({'slices_2d_axial': ..., 'slices_2d_coronal': ..., 'slices_2d_sagittal': ...}),
                             soit directement la liste des coupes axiales.
        :param slices_coronal: La liste des coupes coronales (si non fournie via le dictionnaire).
        :param slices_sagittal: La liste des coupes sagittales (si non fournie via le dictionnaire).
        :param target_num_slices: Nombre de coupes cible par vue (défaut: 20).
        :return: Probabilité moyenne (float) sur l'ensemble des K folds.
        """
        # Unpack du dictionnaire d'entrée pipelines
        if isinstance(slices_input, dict):
            slices_axial = slices_input.get("slices_2d_axial")
            slices_coronal = slices_input.get("slices_2d_coronal")
            slices_sagittal = slices_input.get("slices_2d_sagittal")
        else:
            slices_axial = slices_input

        if slices_axial is None or slices_coronal is None or slices_sagittal is None:
            raise ValueError("Les coupes axiales, coronales et sagittales doivent toutes être fournies pour l'inférence Tri-View.")

        # Preprocessing axial, coronal et sagittal
        batch_ax = self._preprocess_slice_sequence(
            slices_axial, target_num_slices=target_num_slices, index_key="z_index"
        )
        batch_cor = self._preprocess_slice_sequence(
            slices_coronal, target_num_slices=target_num_slices, index_key="y_index"
        )
        batch_sag = self._preprocess_slice_sequence(
            slices_sagittal, target_num_slices=target_num_slices, index_key="x_index"
        )

        if batch_ax is None or batch_cor is None or batch_sag is None:
            return 0.5

        # Forward pass & agrégation
        fold_probs = []
        with torch.no_grad():
            for model in self.models:
                logits = model(batch_ax, batch_cor, batch_sag).squeeze(-1)
                prob = torch.sigmoid(logits).cpu().item()
                fold_probs.append(prob)

        return float(np.mean(fold_probs))


# -----------------------------------------------------------------------------
# 3. Classes d'Ensemble Spécifiques
# -----------------------------------------------------------------------------
class SeResNext26DLSTMSagcoraxEnsemble(BaseEnsembleSagCorAx2DRNN):
    def __init__(self, models_dir, prefix=None, num_folds=5, device=None):
        super().__init__(
            backbone_name="seresnext26d_32x4d",
            rnn_type="lstm",
            rnn_hidden=128,
            models_dir=models_dir,
            prefix=prefix,
            num_folds=num_folds,
            device=device,
        )

class SeResNext26TSGRUSagcoraxEnsemble(BaseEnsembleSagCorAx2DRNN):
    def __init__(self, models_dir, prefix=None, num_folds=5, device=None):
        super().__init__(
            backbone_name="seresnext26ts",
            rnn_type="gru",
            rnn_hidden=128,
            models_dir=models_dir,
            prefix=prefix,
            num_folds=num_folds,
            device=device,
        )

class SeResNext26TSLSTMSagcoraxEnsemble(BaseEnsembleSagCorAx2DRNN):
    def __init__(self, models_dir, prefix=None, num_folds=5, device=None):
        super().__init__(
            backbone_name="seresnext26ts",
            rnn_type="lstm",
            rnn_hidden=128,
            models_dir=models_dir,
            prefix=prefix,
            num_folds=num_folds,
            device=device,
        )

class SeResNext26DGRUSagcoraxEnsemble(BaseEnsembleSagCorAx2DRNN):
    def __init__(self, models_dir, prefix=None, num_folds=5, device=None):
        super().__init__(
            backbone_name="seresnext26d_32x4d",
            rnn_type="gru",
            rnn_hidden=128,
            models_dir=models_dir,
            prefix=prefix,
            num_folds=num_folds,
            device=device,
        )

class Resnet18DLSTMSagcoraxEnsemble(BaseEnsembleSagCorAx2DRNN):
    def __init__(self, models_dir, prefix=None, num_folds=5, device=None):
        super().__init__(
            backbone_name="resnet18d",
            rnn_type="lstm",
            rnn_hidden=128,
            models_dir=models_dir,
            prefix=prefix,
            num_folds=num_folds,
            device=device,
        )