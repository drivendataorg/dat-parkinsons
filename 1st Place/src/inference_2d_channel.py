"""
inference_2d_channel.py
========================
Classe d'inférence orientée objet pour les modèles 2D multi-canaux (Slices comme canaux).
Utilise `timm.create_model` pour charger dynamiquement les architectures (ResNet, EfficientNet, SEResNeXt, etc.).
Encapsule le chargement des K-folds et la prédiction.
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import timm
from PIL import Image
from torchvision import transforms


class BaseEnsemble2DChannel:
    """
    Classe de base factorisée pour charger un ensemble de modèles 2D multi-canaux
    (où chaque tranche/slice est un canal d'entrée) et moyenner leurs prédictions.
    """

    def __init__(
        self,
        model_name,
        models_dir,
        in_chans=5,
        prefix=None,
        num_folds=5,
        img_size=(128, 128),
        device=None,
    ):
        """
        :param model_name: Nom de l'architecture timm (ex: 'seresnext50_32x4d', 'efficientnet_b0')
        :param models_dir: Dossier contenant les fichiers .pth
        :param in_chans: Nombre de tranches/slices utilisées comme canaux d'entrée (target_num_slices)
        :param prefix: Préfixe du nom de fichier des poids
        :param num_folds: Nombre de folds dans l'ensemble
        :param img_size: Taille cible pour le redimensionnement (H, W)
        :param device: Appareil de calcul ('cuda' ou 'cpu')
        """
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.models = []
        self.model_name = model_name
        self.in_chans = in_chans
        self.prefix = prefix if prefix else self.model_name
        self.img_size = img_size

        print(
            f"⚙️ Chargement de l'ensemble 2D Multi-Canaux ({self.model_name}, {in_chans} canaux) "
            f"[Préfixe: '{self.prefix}', {num_folds} folds] sur {self.device}"
        )

        for fold in range(num_folds):
            model_path = os.path.join(models_dir, f"{self.prefix}_fold_{fold}.pth")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Poids introuvables : {model_path}")

            # 1. Instanciation de l'architecture via timm
            model = self._instantiate_model()

            # 2. Chargement sécurisé des poids
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
        Instancie le modèle timm avec le nombre de canaux d'entrée et 1 classe de sortie.
        """
        return timm.create_model(
            self.model_name,
            pretrained=False,
            in_chans=self.in_chans,
            num_classes=1,
        )

    def predict(self, input_data):
        """
        Inférence sur la séquence de tranches du patient.

        :param input_data: Liste de dicts [{'z_index': ..., 'image': np.array}, ...]
                           OU PyTorch Tensor de forme (1, C, H, W) / (C, H, W).
        :return: Probabilité moyenne (float) sur l'ensemble des K folds
        """
        if input_data is None or len(input_data) == 0:
            return 0.5

        # ---------------------------------------------------------------------
        # 1. Conversion si input_data est une Liste de dicts / arrays
        # ---------------------------------------------------------------------
        if isinstance(input_data, list):
            channel_tensors = []
            for item in input_data:
                # Récupération de l'image 2D
                img_gray = item["image"] if isinstance(item, dict) else item

                # Redimensionnement vers la taille cible img_size
                if img_gray.shape[:2] != self.img_size:
                    img_gray = cv2.resize(
                        img_gray, self.img_size, interpolation=cv2.INTER_LINEAR
                    )

                # Convertir en Tensor Float 2D (H, W)
                tensor_2d = torch.from_numpy(img_gray).float()

                # Normalisation locale standard [0, 1] si nécessaire
                if tensor_2d.max() > 1.0:
                    tensor_2d = tensor_2d / 255.0

                channel_tensors.append(tensor_2d)

            # Empilage le long de la dimension des canaux -> (C_actual, H, W)
            image_tensor = torch.stack(channel_tensors, dim=0)

        elif isinstance(input_data, torch.Tensor):
            image_tensor = input_data
        else:
            raise TypeError(f"Type d'entrée non supporté : {type(input_data)}")

        # ---------------------------------------------------------------------
        # 2. Ajustement du nombre de canaux (Padding / Troncature vers in_chans)
        # ---------------------------------------------------------------------
        current_chans = image_tensor.shape[0] if image_tensor.dim() == 3 else image_tensor.shape[1]

        if current_chans != self.in_chans:
            if image_tensor.dim() == 3:  # (C, H, W)
                if current_chans < self.in_chans:
                    pad = torch.zeros((self.in_chans - current_chans, *image_tensor.shape[1:]), dtype=image_tensor.dtype)
                    image_tensor = torch.cat([image_tensor, pad], dim=0)
                else:
                    image_tensor = image_tensor[:self.in_chans, ...]
            elif image_tensor.dim() == 4:  # (B, C, H, W)
                if current_chans < self.in_chans:
                    pad = torch.zeros((image_tensor.shape[0], self.in_chans - current_chans, *image_tensor.shape[2:]), dtype=image_tensor.dtype)
                    image_tensor = torch.cat([image_tensor, pad], dim=1)
                else:
                    image_tensor = image_tensor[:, :self.in_chans, ...]

        # ---------------------------------------------------------------------
        # 3. Formatage de la dimension Batch -> (1, C, H, W)
        # ---------------------------------------------------------------------
        if image_tensor.dim() == 3:
            batch = image_tensor.unsqueeze(0).to(self.device)
        elif image_tensor.dim() == 4:
            batch = image_tensor.to(self.device)
        else:
            raise ValueError(f"Shape inattendue pour image_tensor : {image_tensor.shape}")

        # ---------------------------------------------------------------------
        # 4. Inférence sur les K folds
        # ---------------------------------------------------------------------
        fold_probs = []
        with torch.no_grad():
            for model in self.models:
                outputs = model(batch).squeeze(-1)
                # Conversion logit -> probabilité
                prob = torch.sigmoid(outputs).cpu().item()
                fold_probs.append(prob)

        return float(np.mean(fold_probs))


# ==============================================================================
# CLASSES SPÉCIFIQUES (Héritent de BaseEnsemble2DChannel)
# ==============================================================================

class SeResNext26D2DChannelEnsemble(BaseEnsemble2DChannel):
    def __init__(self, models_dir, in_chans=5, prefix="seresnext26d_32x4d", num_folds=5, device=None):
        super().__init__(
            model_name="seresnext26d_32x4d",
            models_dir=models_dir,
            in_chans=in_chans,
            prefix=prefix,
            num_folds=num_folds,
            device=device,
        )

class SeResNext26TS2DChannelEnsemble(BaseEnsemble2DChannel):
    def __init__(self, models_dir, in_chans=5, prefix="seresnext26ts", num_folds=5, device=None):
        super().__init__(
            model_name="seresnext26ts",
            models_dir=models_dir,
            in_chans=in_chans,
            prefix=prefix,
            num_folds=num_folds,
            device=device,
        )

class RegnetY0022DChannelEnsemble(BaseEnsemble2DChannel):
    def __init__(self, models_dir, in_chans=5, prefix="regnety_002", num_folds=5, device=None):
        super().__init__(
            model_name="regnety_002",
            models_dir=models_dir,
            in_chans=in_chans,
            prefix=prefix,
            num_folds=num_folds,
            device=device,
        )