"""
inference_3d.py
===============
Classe d'inférence orientée objet pour les modèles 3D.
Utilise builder.py pour charger dynamiquement les architectures 
(DenseNet, MedicalNet, MedNeXt, SwinTransformer, EfficientNet).
Encapsule le chargement des K-folds et la prédiction.
"""

import os
import re
import numpy as np
import torch
import monai.transforms as mt

from .builder import build_model, build_model_medicalnet

class BaseEnsemble3D:
    """
    Classe de base factorisée pour charger un ensemble de modèles 3D et moyenner leurs prédictions.
    """
    def __init__(self, model_name, models_dir, prefix=None, num_folds=5, device=None):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.models = []
        self.model_name = model_name.lower()
        self.prefix = prefix if prefix else self.model_name
        
        print(f"⚙️ Chargement de l'ensemble {self.model_name} (Préfixe: '{self.prefix}', {num_folds} folds) sur {self.device}")
        
        for fold in range(num_folds):
            model_path = os.path.join(models_dir, f"{self.prefix}_fold_{fold}.pth")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Poids introuvables : {model_path}")
            
            # 1. Instanciation de l'architecture via builder.py
            model = self._instantiate_model()
            
            # 2. Chargement sécurisé des poids
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Gestion des différentes structures de dictionnaires PyTorch
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
                
            model.load_state_dict(state_dict, strict=False)
            model.to(self.device)
            model.eval()  # Mode inférence (désactive le dropout)
            
            self.models.append(model)
            
    def _instantiate_model(self):
        """
        Instancie l'architecture en appelant le builder.
        Contient un patch pour MedicalNet qui nécessite des arguments supplémentaires.
        """
        name = self.model_name.lower()
        if any(k in name for k in ['medicalnet', 'resnext', 'resnet']):

            if "se-resnext" in name:
                backbone = "se_resnext"
            elif "attn-resnext" in name:
                backbone = "attn_resnext"
            elif "resnext" in name:
                backbone = "resnext"
            elif 'medicalnet' in name:
                backbone = "resnet"
            else:
                raise Exception(f"Backbone unknown for {name}")


            # 2. Déduction de la profondeur
            # Capture les chiffres accolés à resnext/resnet (ex: resnext50)
            depth_match = re.search(r'(?:medicalnet|resnext|resnet)(\d+)', name)
            
            if depth_match:
                depth = int(depth_match.group(1))
            else:
                raise Exception(f"Depth unknown for {name}")
                

            # 3. Extraction des options supplémentaires via kwargs
            extra_kwargs = {}

            # Extraction de la cardinalité (ex: card128 -> 128)
            card_match = re.search(r'card(\d+)', self.model_name)
            if card_match:
                extra_kwargs['resnext_cardinality'] = int(card_match.group(1))
            
            model = build_model_medicalnet(depth=depth, backbone=backbone, **extra_kwargs)
            return model
            
        # Pour tous les autres modèles (MedNeXt, Swin, DenseNet, EfficientNet...)
        return build_model(self.model_name)

    def predict(self, volume_3d):
        """
        Inférence sur le tenseur 3D et agrégation des résultats de l'ensemble.
        """
        if volume_3d is None:
            return 0.5 
            
        # Ajout de la dimension batch si nécessaire -> (1, 1, X, Y, Z)
        if volume_3d.dim() == 4:
            batch = volume_3d.unsqueeze(0).to(self.device)
        elif volume_3d.dim() == 5:
            batch = volume_3d.to(self.device)
        else:
            raise ValueError(f"Shape inattendue pour volume_3d : {volume_3d.shape}. 4 ou 5 dimensions requises.")
        
        fold_probs = []
        with torch.no_grad():
            for model in self.models:
                outputs = model(batch)
                prob = torch.sigmoid(outputs).cpu().item()
                fold_probs.append(prob)
                
        # Retourne la moyenne des probabilités des K folds
        return float(np.mean(fold_probs))

# ==============================================================================
# CLASSES SPÉCIFIQUES (Héritent de BaseEnsemble3D pour simplicité et rétrocompatibilité)
# ==============================================================================

class DenseNet121Ensemble(BaseEnsemble3D):
    def __init__(self, models_dir, prefix="densenet3d", num_folds=5, device=None):
        super().__init__(model_name="densenet121", models_dir=models_dir, 
                         prefix=prefix, num_folds=num_folds, device=device)

class DenseNet201Ensemble(BaseEnsemble3D):
    def __init__(self, models_dir, prefix="densenet201", num_folds=5, device=None):
        super().__init__(model_name="densenet201", models_dir=models_dir, 
                         prefix=prefix, num_folds=num_folds, device=device)

class AttentionDenseNet121Ensemble(BaseEnsemble3D):
    def __init__(self, models_dir, prefix="densenet3d_attn", num_folds=5, device=None):
        # Le model_name contient "attention" pour déclencher le wrapper dans builder.py
        # Le prefix par défaut correspond au nom de fichier dans ton script d'entraînement (densenet3d_attn_fold_X.pth)
        super().__init__(model_name="densenet121_attention", models_dir=models_dir, 
                         prefix=prefix, num_folds=num_folds, device=device)

class AttentionDenseNet201Ensemble(BaseEnsemble3D):
    def __init__(self, models_dir, prefix="densenet201_attn", num_folds=5, device=None):
        super().__init__(model_name="densenet201_attention", models_dir=models_dir, 
                         prefix=prefix, num_folds=num_folds, device=device)

class MedicalNetEnsemble(BaseEnsemble3D):
    def __init__(self, models_dir, prefix="medicalnet10", num_folds=5, device=None):
        super().__init__(model_name=prefix, models_dir=models_dir, 
                         prefix=prefix, num_folds=num_folds, device=device)

class MedNeXtEnsemble(BaseEnsemble3D):
    def __init__(self, models_dir, prefix="mednext", num_folds=5, device=None):
        super().__init__(model_name=prefix, models_dir=models_dir, 
                         prefix=prefix, num_folds=num_folds, device=device)

class SwinTFEnsemble(BaseEnsemble3D):
    def __init__(self, models_dir, prefix="swintf", num_folds=5, device=None):
        super().__init__(model_name="swintf", models_dir=models_dir, 
                         prefix=prefix, num_folds=num_folds, device=device)

class EfficientNetEnsemble(BaseEnsemble3D):
    def __init__(self, models_dir, prefix="efficientnet-b0", num_folds=5, device=None):
        # Permet de supporter "efficientnet-b0", "efficientnet-b1", etc. selon le prefix
        super().__init__(model_name=prefix, models_dir=models_dir, 
                         prefix=prefix, num_folds=num_folds, device=device)

class MiniCNN3DSiameseEnsemble(BaseEnsemble3D):
    def __init__(self, models_dir, prefix="minicnn3d", n_exp=20, num_folds=5, device=None):
        # On ne fait PAS appel à super().__init__() pour éviter la boucle fold simple
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.models = []
        self.model_name = "minicnn3d_siamese"
        self.prefix = prefix
        
        # 🎯 AJOUT : Les mêmes transforms qu'en validation lors de l'entraînement
        self.val_transforms = mt.Compose([
            mt.GaussianSmooth(sigma=1.0),
            mt.ScaleIntensity(),
        ])
        
        print(f"⚙️ Chargement de l'ensemble {self.model_name} (Préfixe: '{self.prefix}', {n_exp} exps x {num_folds} folds) sur {self.device}")
        loaded_count = 0

        # Double boucle pour charger le format xxx_exp_X_fold_Y.pth
        for exp in range(n_exp):
            for fold in range(num_folds):
                model_path = os.path.join(models_dir, f"{self.prefix}_exp{exp}_fold{fold}.pth")
                
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"Poids introuvables : {model_path}")
                
                model = self._instantiate_model()
                checkpoint = torch.load(model_path, map_location=self.device)
                
                if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                else:
                    state_dict = checkpoint
                    
                model.load_state_dict(state_dict, strict=True)
                model.to(self.device)
                model.eval()
                
                self.models.append(model)
                loaded_count += 1
                
        expected = n_exp * num_folds
        if loaded_count != expected:
            raise FileNotFoundError(
                f"{prefix} : {loaded_count}/{expected} modèles chargés. "
                f"La température est calibrée pour {expected}."
            )
        print(f"✅ {loaded_count} modèles chargés avec succès en VRAM.")

    def predict(self, siamese_cubes_dict):
        """ Surcharge de la méthode predict pour traiter cube_L et cube_R """
        if siamese_cubes_dict is None:
            return 0.5 
            
        # 1. On récupère les tenseurs bruts (sur CPU car venant de preprocessing.py)
        # Shape d'origine dans le preprocessing : (1, 32, 32, 32)
        cube_L_raw = siamese_cubes_dict["cube_L"]
        cube_R_raw = siamese_cubes_dict["cube_R"]
        
        # 🎯 2. CORRECTION DU BUG : On recrée le "combo" pour appliquer la normalisation conjointe
        # Concaténation sur la dimension Channel -> (2, 32, 32, 32)
        combo_3d = torch.cat([cube_L_raw, cube_R_raw], dim=0)
        
        # Application des transforms (Lissage + Min-Max global préservant le ratio G/D)
        combo_3d = self.val_transforms(combo_3d)
        
        # 3. On sépare de nouveau les cubes transformés et on les envoie sur le GPU
        cube_L = combo_3d[0:1].to(self.device)  # (1, 32, 32, 32)
        cube_R = combo_3d[1:2].to(self.device)  # (1, 32, 32, 32)
        
        # 4. Ajout de la dimension de batch -> (1, 1, 32, 32, 32)
        if cube_L.dim() == 4:
            cube_L = cube_L.unsqueeze(0)
            cube_R = cube_R.unsqueeze(0)
            
        fold_probs = []
        with torch.no_grad():
            for model in self.models:
                outputs = model(cube_L, cube_R)
                prob = torch.sigmoid(outputs).cpu().item()
                fold_probs.append(prob)
                
        return float(np.mean(fold_probs))