import sys
import subprocess
import argparse
from pathlib import Path
from loguru import logger
from omegaconf import OmegaConf

# ==========================================
# PHASE ORDER (For the --start-at argument)
# ==========================================
PHASES = [
    "phase1_prep",
    "phase1_train",
    "phase2",
    "phase3",
    "phase4_prep",
    "phase4_train",
]

# ==========================================
# PATH CONFIGURATION (Scripts without arguments)
# ==========================================
SCRIPTS = {
    "phase1_prep_folds": "1_initial_training/prepare_folds.py",
    "phase1_prep_2d": "1_initial_training/prepare_2d_dataset.py",
    "train_svm": "1_initial_training/train_svm.py",
    "phase2_stacking": "2_stacking/stacking.py",
    "phase3_soft_labeling": "3_soft_labeling/soft_labeling.py",
    "phase4_prep_siamese": "4_siamese_training/prepare_2d_dataset_siamese.py",
    "phase4_prep_striatums": "4_siamese_training/prepare_2d_dataset_striatum.py",
    "phase4_training": "4_siamese_training/train_siamese.py",
}

# ==========================================
# MODEL COMMANDS (Dynamic)
# ==========================================
MODEL_TO_CMD = {
    "ResNet2D": [sys.executable, "1_initial_training/train_2d_resnet.py"],
    "Vit2D": [sys.executable, "1_initial_training/train_2d_vit.py"],
    "DenseNet3D121": [sys.executable, "1_initial_training/train_3d_densenet121.py"],
    "DenseNet3D201": [sys.executable, "1_initial_training/train_3d_densenet201.py"],
    "DenseNetAttention": [sys.executable, "1_initial_training/train_3d_attention.py"],
    "MedicalNet18": [sys.executable, "1_initial_training/train_medicalnet.py", "--run_name", "medicalnet18", "--model", "medicalnet", "--medicalnet_backbone", "resnet", "--medicalnet_depth", "18", "--resnet_shortcut", "A"],
    "MedicalNet34": [sys.executable, "1_initial_training/train_medicalnet.py", "--run_name", "medicalnet34", "--model", "medicalnet", "--medicalnet_backbone", "resnet", "--medicalnet_depth", "34", "--resnet_shortcut", "A"],
    "MedicalNet50": [sys.executable, "1_initial_training/train_medicalnet.py", "--run_name", "medicalnet50", "--model", "medicalnet", "--medicalnet_backbone", "resnet", "--medicalnet_depth", "50", "--resnet_shortcut", "B"],
    "MedicalNet200": [sys.executable, "1_initial_training/train_medicalnet.py", "--run_name", "medicalnet200", "--model", "medicalnet", "--medicalnet_backbone", "resnet", "--medicalnet_depth", "200", "--resnet_shortcut", "B"],
    "MedicalNetResNext50": [sys.executable, "1_initial_training/train_medicalnet.py", "--run_name", "medicalnet-resnext50", "--model", "medicalnet", "--medicalnet_backbone", "resnext", "--medicalnet_depth", "50", "--resnet_shortcut", "B", "--resnext_cardinality", "32"],
    "MedicalNetResNext101": [sys.executable, "1_initial_training/train_medicalnet.py", "--run_name", "medicalnet-resnext101", "--model", "medicalnet", "--medicalnet_backbone", "resnext", "--medicalnet_depth", "101", "--resnet_shortcut", "B", "--resnext_cardinality", "32"],
    "MedNextB": [sys.executable, "1_initial_training/train_medicalnet.py", "--run_name", "medicalnextB", "--model", "mednext"],
}

def run_command(name: str, cmd: list) -> None:
    logger.info(f"▶️ STARTING: {name}")
    logger.info(f"💻 Command: {' '.join(cmd)}")
    try:
        subprocess.check_call(cmd)
        logger.success(f"✅ SUCCESS: {name}\n" + "-"*80)
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ FATAL ERROR: Step '{name}' failed with return code {e.returncode}.")
        sys.exit(1)
    except FileNotFoundError:
        logger.error(f"❌ FATAL ERROR: Script '{cmd[1]}' not found.")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Global orchestrator for the Parkinson's ML pipeline")
    parser.add_argument(
        "--start-at", 
        type=str, 
        choices=PHASES, 
        default="phase1_prep", 
        help="Phase from which to start the pipeline."
    )
    parser.add_argument(
        "--no-synth", 
        action="store_true", 
        help="Deactivate the use of synthetic data (skips phases 1-3 automatically and modifies phase 4)."
    )
    args = parser.parse_args()

    start_idx = PHASES.index(args.start_at)

    logger.info("🚀 LAUNCHING GLOBAL DAT-PARK PIPELINE")
    
    # Logique d'ajustement automatique pour --no-synth
    if args.no_synth:
        logger.warning("🚫 SYNTHETIC DATA DEACTIVATED (--no-synth)")
        if start_idx < PHASES.index("phase4_prep"):
            logger.info("⏩ Automatically skipping phases 1 to 3. Starting at 'phase4_prep'.")
            start_idx = PHASES.index("phase4_prep")
    else:
        logger.info(f"📍 Starting point: {args.start_at}")

    print("="*80)
    cfg = OmegaConf.load("0_configs/default.yaml")

    # ---------------------------------------------------------
    # PHASE 1: PREPARATION
    # ---------------------------------------------------------
    if start_idx <= PHASES.index("phase1_prep"):
        run_command("PHASE 1 - Folds Preparation", [sys.executable, SCRIPTS["phase1_prep_folds"]])
        run_command("PHASE 1 - 2D Dataset Preparation", [sys.executable, SCRIPTS["phase1_prep_2d"]])
    else:
        logger.warning("⏭️ Phase 1 (Preparation) skipped.")

    # ---------------------------------------------------------
    # PHASE 1: INITIAL DYNAMIC TRAINING
    # ---------------------------------------------------------
    if start_idx <= PHASES.index("phase1_train"):
        active_models = cfg.get("stacking", {}).get("active_models", [])
        for model_name in active_models:
            if model_name in MODEL_TO_CMD:
                cmd = MODEL_TO_CMD[model_name]
                if Path(cmd[1]).exists():
                    run_command(f"PHASE 1 - Train {model_name}", cmd)
        
        tabular_oof = cfg.get("stacking", {}).get("tabular_oof", None)
        if tabular_oof is not None and str(tabular_oof).lower() != "null":
            run_command("PHASE 1 - Train SVM", [sys.executable, SCRIPTS["train_svm"]])
    else:
        logger.warning("⏭️ Phase 1 (Initial Training) skipped.")

    # ---------------------------------------------------------
    # PHASE 2: STACKING
    # ---------------------------------------------------------
    if start_idx <= PHASES.index("phase2"):
        run_command("PHASE 2 - Stacking", [sys.executable, SCRIPTS["phase2_stacking"]])
    else:
        logger.warning("⏭️ Phase 2 (Stacking) skipped.")

    # ---------------------------------------------------------
    # PHASE 3: SOFT LABELING
    # ---------------------------------------------------------
    if start_idx <= PHASES.index("phase3"):
        run_command("PHASE 3 - Synthetic Soft Labeling", [sys.executable, SCRIPTS["phase3_soft_labeling"]])
    else:
        logger.warning("⏭️ Phase 3 (Soft Labeling) skipped.")

    # ---------------------------------------------------------
    # PHASE 4: SIAMESE PREP
    # ---------------------------------------------------------
    if start_idx <= PHASES.index("phase4_prep"):
        run_command("PHASE 4 PREP - Siamese (REAL)", [sys.executable, SCRIPTS["phase4_prep_siamese"], "--mode", "real"])
        run_command("PHASE 4 PREP - Striatums (REAL)", [sys.executable, SCRIPTS["phase4_prep_striatums"], "--mode", "real"])
        
        if not args.no_synth:
            run_command("PHASE 4 PREP - Siamese (SYNTHETIC)", [sys.executable, SCRIPTS["phase4_prep_siamese"], "--mode", "synthetic"])
            run_command("PHASE 4 PREP - Striatums (SYNTHETIC)", [sys.executable, SCRIPTS["phase4_prep_striatums"], "--mode", "synthetic"])
        else:
            logger.info("🚫 Phase 4 Prep - Synthetic data generation skipped (--no-synth).")
    else:
        logger.warning("⏭️ Phase 4 (Tensor Preparation) skipped.")

    # ---------------------------------------------------------
    # PHASE 4: SIAMESE TRAINING
    # ---------------------------------------------------------
    if start_idx <= PHASES.index("phase4_train"):
        train_cmd = [sys.executable, SCRIPTS["phase4_training"]]
        if args.no_synth:
            train_cmd.append("--no-synth")  # On passe l'argument au script !
        run_command("PHASE 4 - Multi-View Siamese Training", train_cmd)
    else:
        logger.warning("⏭️ Phase 4 (Siamese Training) skipped.")

    logger.success("🎉 ALL TARGETED PHASES HAVE BEEN EXECUTED SUCCESSFULLY! 🎉")

if __name__ == "__main__":
    main()