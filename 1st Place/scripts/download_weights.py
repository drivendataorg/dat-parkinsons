import os
import sys
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from loguru import logger
from omegaconf import OmegaConf
from huggingface_hub import snapshot_download

# ==========================================
# MAPPING TO CORRECT FOLDER NAMES
# ==========================================
HF_TO_YAML_MAPPING = {
    "mednext-B": "medicalnextB"
}

def backup_if_exists(target_path: Path):
    if target_path.exists() and any(target_path.iterdir()):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = target_path.parent / f"{target_path.name}_backup_{timestamp}"
        logger.warning(f"⚠️ Directory '{target_path}' already exists!")
        logger.info(f"🔄 Backing up to: {backup_dir}")
        shutil.move(str(target_path), str(backup_dir))
        logger.success("✅ Backup complete.")

def main():
    logger.info("🚀 STARTING WEIGHTS DOWNLOAD (Hugging Face)")
    print("="*80)

    try:
        cfg = OmegaConf.load("0_configs/default.yaml")
    except FileNotFoundError:
        logger.error("❌ Could not find '0_configs/default.yaml'.")
        sys.exit(1)

    phase1_out = Path(cfg.paths.phase1_out)
    phase3_out = Path(cfg.paths.phase3_out)
    phase4_out = Path(cfg.paths.phase4_out)

    # Security backups
    backup_if_exists(phase1_out)
    backup_if_exists(phase3_out)
    backup_if_exists(phase4_out)

    phase1_out.mkdir(parents=True, exist_ok=True)
    phase3_out.mkdir(parents=True, exist_ok=True)
    phase4_out.mkdir(parents=True, exist_ok=True)

    logger.info("⬇️ Downloading repository from Hugging Face...")
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        repo_path = snapshot_download(
            repo_id="dinhviettoanle/dat-parkinson-models", 
            local_dir=tmp_dir
        )
        
        repo_root = Path(repo_path)
        
        if (repo_root / "models").exists():
            base_dir = repo_root / "models"
        else:
            base_dir = repo_root
            
        hf_phase1 = base_dir / "01-soft_labeling"
        hf_phase4 = base_dir / "02-siamese"

        print("="*80)
        logger.info("📦 DISPATCHING PHASE 1 & 3 WEIGHTS")
        
        if hf_phase1.exists():
            for item in hf_phase1.iterdir():
                if item.is_file():
                    if item.name in ["SVM.pkl", "tabular_oof.csv"]:
                        # Phase 1: SVM and Tabular OOF
                        svm_folder = phase1_out / cfg.models.SVM.folder
                        svm_folder.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(item, svm_folder / item.name)
                        logger.info(f" ├── Copied: {item.name} -> {phase1_out.name}/{svm_folder.name}/")
                    
                    elif item.name == "synthetic_soft_labels.csv":
                        # Phase 3: Synthetic soft labels
                        shutil.copy2(item, phase3_out / "synthetic_soft_labels.csv")
                        logger.info(f" ├── Copied: synthetic_soft_labels.csv -> {phase3_out.name}/")
                
                elif item.is_dir():
                    # Phase 1: Model folders (ResNet, MedicalNet, etc.)
                    target_name = HF_TO_YAML_MAPPING.get(item.name, item.name)
                    target_dir = phase1_out / target_name
                    shutil.copytree(item, target_dir, dirs_exist_ok=True)
                    
                    # --- SPECIFIC MEDNEXT CORRECTIONS ---
                    if target_name == "medicalnextB":
                        # 1. Correct CSV name
                        old_csv = target_dir / "oof_mednext-B.csv"
                        new_csv = target_dir / "oof_mednextB.csv"
                        if old_csv.exists():
                            old_csv.rename(new_csv)
                            
                        # 2. Correct weights prefix (mednext-B -> mednext_b)
                        for pth_file in target_dir.glob("mednext-B_fold_*.pth"):
                            new_name = pth_file.name.replace("mednext-B", "mednext_b")
                            pth_file.rename(target_dir / new_name)
                            
                    logger.info(f" ├── Copied: {item.name}/ -> {phase1_out.name}/{target_name}/")
        else:
            logger.warning(f"⚠️ Folder {hf_phase1.name} not found in the repository.")

        logger.info("\n📦 DISPATCHING PHASE 4 WEIGHTS (Siamese)")
        if hf_phase4.exists():
            siamois_target = phase4_out / "siamese_network"
            siamois_target.mkdir(parents=True, exist_ok=True)
            shutil.copytree(hf_phase4, siamois_target, dirs_exist_ok=True)
            num_siamois = len(list(siamois_target.glob("*.pth")))
            logger.info(f" ├── Copied: {num_siamois} siamese models -> {phase4_out.name}/{siamois_target.name}/")
        else:
            logger.warning(f"⚠️ Folder {hf_phase4.name} not found in the repository.")

    print("="*80)
    logger.success("🎉 SUCCESSFULLY COMPLETED ! 🎉")

if __name__ == "__main__":
    main()