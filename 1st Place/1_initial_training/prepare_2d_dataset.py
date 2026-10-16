import os
import cv2
import glob
import numpy as np
import pandas as pd
import concurrent.futures
import multiprocessing
from tqdm import tqdm
from omegaconf import OmegaConf
from pathlib import Path
from src.preprocessing import preprocess_patient, extract_2d_slices


# ==========================================
# CONFIGURATION
# ==========================================
cfg = OmegaConf.merge(
    OmegaConf.load("0_configs/default.yaml"), 
    OmegaConf.from_cli()
)

DATA_DIR    = cfg.paths.niftis
LABELS_PATH = cfg.paths.train_labels

OUTPUT_DIR  = cfg.paths.dataset_2d
IMG_OUT_DIR = cfg.paths.images_2d
CSV_OUT     = cfg.paths.folds_2d_csv

NUM_SLICES = 5 # Nombre maximum de coupes tentées par patient
STEP_Z = 2     # Espacement entre les coupes (ex: z-4, z-2, z, z+2, z+4)

# Seuils pour rejeter les images trop noires
MIN_MAX_INTENSITY = 40  
MIN_MEAN_INTENSITY = 1.0 

os.makedirs(IMG_OUT_DIR, exist_ok=True)


# ==========================================
# WORKER POUR LE MULTIPROCESSING
# ==========================================
def _process_patient(args):
    uid, label, fold, img_path = args
    records = []
    
    if not os.path.exists(img_path):
        return records
        
    # --- LOGIQUE DE REPRISE (RESUMABILITY) ---
    existing_files = glob.glob(os.path.join(IMG_OUT_DIR, f"{uid}_z*.png"))
    if len(existing_files) > 0:
        for filepath in existing_files:
            filename = os.path.basename(filepath)
            z_str = filename.replace(f"{uid}_z", "").replace(".png", "")
            records.append({
                'uid': uid,
                'slice_path': filepath,
                'label': label,
                'fold': fold,
                'z_index': int(z_str)
            })
        return records

    # --- SI NON PRÉSENT, ON APPELLE PREPROCESSING.PY ---
    try:
        # 1. Prétraitement 3D (resample, crop centré sur le striatum, normalisation)
        # Utilise la fonction unique de ton fichier preprocessing.py
        base_data = preprocess_patient(img_path)
        vol = base_data["volume"]
        
        # 2. Extraction des coupes 2D 
        # (Utilise la logique du masque gaussien que tu as ajoutée dans preprocessing.py)
        slices = extract_2d_slices(
            volume=vol, 
            num_slices=NUM_SLICES, 
            step_z=STEP_Z,
            min_max_intensity=MIN_MAX_INTENSITY, 
            min_mean_intensity=MIN_MEAN_INTENSITY
        )
        
        # 3. Sauvegarde sur disque (PNG)
        for s in slices:
            z_safe = s["z_index"]
            slice_img = s["image"]
            
            filename = f"{uid}_z{z_safe}.png"
            out_filepath = os.path.join(IMG_OUT_DIR, filename)
            cv2.imwrite(out_filepath, slice_img)
            
            records.append({
                'uid': uid,
                'slice_path': out_filepath,
                'label': label,
                'fold': fold,
                'z_index': z_safe
            })
            
    except Exception as e:
        print(f"⚠️ Erreur sur {uid}: {e}")
        
    return records


# ==========================================
# GESTIONNAIRE PRINCIPAL
# ==========================================
def main():
    print(f"🚀 Génération du Dataset 2.5D (Tentative de {NUM_SLICES} coupes max/patient)...")
    
    df = pd.read_csv(LABELS_PATH)
    
    if 'fold' not in df.columns:
        raise ValueError(f"⚠️ Le fichier {LABELS_PATH} ne contient pas de colonne 'fold'.")

    tasks = []
    for _, row in df.iterrows():
        uid = str(row['uid'])
        img_path = os.path.join(DATA_DIR, f"{uid}.nii.gz")
        tasks.append((uid, row['is_pathologic'], row['fold'], img_path))
        
    records = []
    num_workers = min(8, multiprocessing.cpu_count())
    print(f"⚙️ Traitement Multi-Core activé : {num_workers} cœurs alloués.")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as pool:
        for patient_records in tqdm(pool.map(_process_patient, tasks), total=len(tasks)):
            records.extend(patient_records)

    df_2d = pd.DataFrame(records)
    
    if not df_2d.empty:
        df_2d.sort_values(by=['uid', 'z_index'], inplace=True)
        
    df_2d.to_csv(CSV_OUT, index=False)
    
    print(f"\n✅ Terminé ! {len(df_2d)} images 2D générées dans '{IMG_OUT_DIR}'.")
    
    if not df_2d.empty:
        avg_slices = df_2d.groupby('uid').size().mean()
        print(f"✅ Moyenne de coupes valides conservées par patient : {avg_slices:.2f} / {NUM_SLICES}")
        
    print(f"✅ Fichier de mapping : '{CSV_OUT}'.")
    print(df_2d['fold'].value_counts().sort_index())

if __name__ == '__main__':
    main()