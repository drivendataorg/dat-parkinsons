"""
inference_2d_rnn.py
===================
Classe d'inférence orientée objet pour les modèles 2D CNN + RNN (Sequence).
Encapsule le chargement des K-folds et la prédiction pour les modèles 2D-RNN.
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import timm

# -----------------------------------------------------------------------------
# 3. Model Architecture: 2D-CNN Backbone + RNN (LSTM or GRU) + Attention
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


class CNN2D_RNN_Model(nn.Module):
    def __init__(self, backbone_name="resnet18d", pretrained=True, rnn_type="lstm", rnn_hidden=128, num_layers=2):
        super().__init__()
        
        # 1. Feature Extractor 2D (Timm)
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            in_chans=3,
            num_classes=0
        )
        
        feature_dim = self.backbone.num_features
        
        # 2. Module Récurrent Dynamique (LSTM ou GRU)
        rnn_type = rnn_type.lower()
        if rnn_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=feature_dim,
                hidden_size=rnn_hidden,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=0.2 if num_layers > 1 else 0.0
            )
        elif rnn_type == "gru":
            self.rnn = nn.GRU(
                input_size=feature_dim,
                hidden_size=rnn_hidden,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=0.2 if num_layers > 1 else 0.0
            )
        else:
            raise ValueError(f"rnn_type inconnu : '{rnn_type}'. Choisissez 'lstm' ou 'gru'.")
        
        rnn_out_dim = rnn_hidden * 2  # car bidirectional=True
        
        # 3. Attention Temporelle & Tête de Classification
        self.attention = TemporalAttention(feature_dim=rnn_out_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(rnn_out_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x shape: (B, S, 3, H, W)
        B, S, C, H, W = x.shape
        
        x_flat = x.view(B * S, C, H, W)
        features_2d = self.backbone(x_flat)  # (B * S, feature_dim)
        
        sequence_input = features_2d.view(B, S, -1)  # (B, S, feature_dim)
        
        # Passage dans le RNN (LSTM renvoie (out, (h, c)), GRU renvoie (out, h))
        rnn_out, _ = self.rnn(sequence_input)        # (B, S, rnn_out_dim)
        
        context_vector = self.attention(rnn_out)     # (B, rnn_out_dim)
        logits = self.classifier(context_vector)     # (B, 1)
        return logits



class BaseEnsemble2DRNN:
    """
    Classe de base factorisée pour charger un ensemble de modèles 2D-CNN + RNN
    et moyenner leurs prédictions.
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

        # Définition du préfixe des fichiers de checkpoint
        self.prefix = prefix if prefix else f"2drnn_{self.backbone_name}_{self.rnn_type}"

        self.transform = transforms.Compose([
            transforms.Resize(img_size, antialias=True),
        ])

        print(
            f"⚙️ Chargement de l'ensemble 2D-RNN ({self.backbone_name} + {self.rnn_type.upper()}) "
            f"[Préfixe: '{self.prefix}', {num_folds} folds] sur {self.device}"
        )

        for fold in range(num_folds):
            model_path = os.path.join(models_dir, f"{self.prefix}_fold_{fold}.pth")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Poids introuvables : {model_path}")

            # 1. Instanciation du modèle CNN+RNN
            model = self._instantiate_model()

            # 2. Chargement des poids
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
            model.eval()  # Mode inférence

            self.models.append(model)

    def _instantiate_model(self):
        """
        Instancie l'architecture CNN2D_RNN_Model.
        """
        return CNN2D_RNN_Model(
            backbone_name=self.backbone_name,
            pretrained=False,
            rnn_type=self.rnn_type,
            rnn_hidden=self.rnn_hidden,
        )

    def predict(self, slices_2d_seq, target_num_slices=20):
        """
        Inférence sur la séquence de coupe du patient.

        :param slices_2d_seq: Liste de dictionnaires [{'z_index': ..., 'image': np.array}, ...]
                              ou PyTorch Tensor.
        :param target_num_slices: Nombre cible de coupes (défaut: 20)
        :return: Probabilité moyenne (float) sur l'ensemble des K folds
        """
        if not slices_2d_seq:
            return 0.5

        if isinstance(slices_2d_seq, list):
            # 1. Sort by z_index if items are dictionaries
            if len(slices_2d_seq) > 0 and isinstance(slices_2d_seq[0], dict) and "z_index" in slices_2d_seq[0]:
                slices_2d_seq = sorted(slices_2d_seq, key=lambda x: x["z_index"])

            # 2. Resample indices using .round().astype(int) (matches Dataset method)
            num_slices = len(slices_2d_seq)
            if num_slices != target_num_slices:
                indices = np.linspace(0, num_slices - 1, target_num_slices).round().astype(int)
                slices_2d_seq = [slices_2d_seq[i] for i in indices]

            slice_tensors = []
            for slice_item in slices_2d_seq:
                img_gray = slice_item["image"] if isinstance(slice_item, dict) else slice_item

                # Ensure 2D numpy array (H, W)
                if img_gray.ndim == 3 and img_gray.shape[-1] in [1, 3]:
                    img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)

                # Float32 conversion and 0-1 scaling (matches Dataset)
                img_float = img_gray.astype(np.float32) / 255.0

                # 1 channel -> 3 identical channels along axis 0: (3, H, W)
                img_3ch = np.stack([img_float] * 3, axis=0)

                # Convert to Tensor directly from Numpy array
                tensor = torch.from_numpy(img_3ch)

                # Apply spatial transforms directly to Tensor (v2.Resize)
                if self.transform:
                    tensor = self.transform(tensor)

                slice_tensors.append(tensor)

            # Stack along temporal dimension: (S, 3, H, W)
            sequence_tensor = torch.stack(slice_tensors, dim=0)

            # Add batch dimension: (1, S, 3, H, W)
            batch = sequence_tensor.unsqueeze(0).to(self.device)

        elif isinstance(slices_2d_seq, torch.Tensor):
            if slices_2d_seq.dim() == 4:
                batch = slices_2d_seq.unsqueeze(0).to(self.device)
            elif slices_2d_seq.dim() == 5:
                batch = slices_2d_seq.to(self.device)
            else:
                raise ValueError(f"Shape inattendue pour le Tensor d'entrée : {slices_2d_seq.shape}")
        else:
            raise TypeError(f"Type d'entrée non supporté : {type(slices_2d_seq)}")

        # ---------------------------------------------------------------------
        # Inférence & Agrégation
        # ---------------------------------------------------------------------
        fold_probs = []
        with torch.no_grad():
            for model in self.models:
                logits = model(batch).squeeze(-1)
                prob = torch.sigmoid(logits).cpu().item()
                fold_probs.append(prob)

        return float(np.mean(fold_probs))


# ==============================================================================
# CLASSES SPÉCIFIQUES (Héritent de BaseEnsemble2DRNN pour plus de simplicité)
# ==============================================================================

class Resnet34DGRUEnsemble(BaseEnsemble2DRNN):
    def __init__(self, models_dir, prefix=None, num_folds=5, device=None):
        super().__init__(
            backbone_name="resnet34d",
            rnn_type="gru",
            rnn_hidden=128,
            models_dir=models_dir,
            prefix=prefix,
            num_folds=num_folds,
            device=device,
        )

class Resnet18DGRUEnsemble(BaseEnsemble2DRNN):
    def __init__(self, models_dir, prefix=None, num_folds=5, device=None):
        super().__init__(
            backbone_name="resnet18d",
            rnn_type="gru",
            rnn_hidden=128,
            models_dir=models_dir,
            prefix=prefix,
            num_folds=num_folds,
            device=device,
        )

class Resnet18DLSTMEnsemble(BaseEnsemble2DRNN):
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

class SeResNext26DGRUEnsemble(BaseEnsemble2DRNN):
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