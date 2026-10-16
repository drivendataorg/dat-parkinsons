import os
import argparse
import pandas as pd
import torch
import concurrent.futures
import multiprocessing
from tqdm import tqdm
from omegaconf import OmegaConf

from src.preprocessing import prepare_all_inputs

# ==========================================
# WORKER POUR LE MULTIPROCESSING
# ==========================================
def _process_patient(args):
    uid, label, fold, img_path, patch_out_dir, invert_y = args
    out_filepath = os.path.join(patch_out_dir, f"{uid}.pt")
    
    # --- LOGIQUE DE REPRISE (RESUMABILITY) ---
    if os.path.exists(out_filepath):
        return {
            'uid': uid,
            'patch_path': out_filepath,
            'label': label,
            'fold': fold
        }
        
    if not os.path.exists(img_path):
        return None

    # --- SI NON PRÉSENT, ON APPELLE PREPROCESSING.PY ---
    try:
        # 1. Prétraitement complet avec l'option INVERT_Y dynamique
        inputs = prepare_all_inputs(img_path, invert_y)
        patch = inputs["striatums"]["patch_global"]
        
        if patch.ndim == 2:
            patch = patch.unsqueeze(0)
                
        # 2. Sauvegarde rapide sur disque au format PyTorch
        torch.save(patch, out_filepath)
        
        return {
            'uid': uid,
            'patch_path': out_filepath,
            'label': label,
            'fold': fold
        }
        
    except Exception as e:
        print(f"⚠️ Erreur sur le patient {uid}: {e}")
        return None


# ==========================================
# GESTIONNAIRE PRINCIPAL
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Générateur de Patchs Striatums")
    parser.add_argument('--mode', type=str, choices=['real', 'synthetic'], required=True)
    args = parser.parse_args()

    print(f"🚀 Génération des patchs globaux (Mode : {args.mode.upper()})...")
    
    # --- CONFIGURATION DYNAMIQUE SELON LE MODE ---
    cfg = OmegaConf.merge(OmegaConf.load("0_configs/default.yaml"))
    
    if args.mode == 'synthetic':
        DATA_DIR      = cfg.paths.synthetic_niftis
        LABELS_PATH   = os.path.join(cfg.paths.phase3_out, 'synthetic_soft_labels.csv')
        OUTPUT_DIR    = cfg.paths.dataset_synthetic_striatums
        INVERT_Y      = True
    else:
        DATA_DIR      = cfg.paths.niftis
        LABELS_PATH   = cfg.paths.train_labels
        OUTPUT_DIR    = cfg.paths.dataset_striatums
        INVERT_Y      = False

    PATCH_OUT_DIR = os.path.join(OUTPUT_DIR, 'patches_pt')
    CSV_OUT       = os.path.join(OUTPUT_DIR, 'folds.csv')
    os.makedirs(PATCH_OUT_DIR, exist_ok=True)

    # --- LECTURE DU DATASET ---
    df = pd.read_csv(LABELS_PATH)
    label_col = 'is_pathologic' if 'is_pathologic' in df.columns else 'label'
    
    has_fold = 'fold' in df.columns
    if has_fold:
        print("📂 Colonne 'fold' détectée dans le CSV.")
    else:
        print("⚠️ Pas de colonne 'fold' détectée, attribution du fold -1 par défaut.")

    tasks = []
    for _, row in df.iterrows():
        uid = str(row['uid'])
        img_path = os.path.join(DATA_DIR, f"{uid}.nii.gz")
        fold_val = row['fold'] if has_fold else -1
        # On passe PATCH_OUT_DIR et INVERT_Y directement au worker
        tasks.append((uid, row[label_col], fold_val, img_path, PATCH_OUT_DIR, INVERT_Y))
        
    records = []
    num_workers = min(48, multiprocessing.cpu_count())
    print(f"⚙️ Traitement Multi-Core activé : {num_workers} cœurs alloués. (Invert Y : {INVERT_Y})")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as pool:
        for res in tqdm(pool.map(_process_patient, tasks), total=len(tasks)):
            if res is not None:
                records.append(res)

    df_dataset = pd.DataFrame(records)
    
    if not df_dataset.empty:
        df_dataset.sort_values(by=['uid'], inplace=True)
        
    df_dataset.to_csv(CSV_OUT, index=False)
    
    print(f"\n✅ Terminé ! {len(df_dataset)} fichiers de patchs générés dans '{PATCH_OUT_DIR}'.")
    print(f"✅ Fichier de mapping sauvegardé : '{CSV_OUT}'.")
    if has_fold:
        print(df_dataset['fold'].value_counts().sort_index())

if __name__ == '__main__':
    main()