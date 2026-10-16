import os
import sys
import shutil
import subprocess
from pathlib import Path
from loguru import logger
from omegaconf import OmegaConf

def run_cmd(cmd, cwd=None):
    """Exécute une commande dans le shell et stoppe en cas d'erreur."""
    logger.info(f"💻 Commande : {' '.join(cmd)}")
    subprocess.check_call(cmd, cwd=cwd)

def main():
    logger.info("🚀 DÉMARRAGE DE LA PHASE 6 : PRÉPARATION ET TEST (DOCKER)")
    
    # ==========================================
    # 1. DÉFINITION DES CHEMINS (CODE VS DATA)
    # ==========================================
    PROJECT_DIR = Path(__file__).resolve().parent.parent
    
    RUNTIME_DIR = PROJECT_DIR / "competition-sfmn-parkinsons-runtime"
    SUB_SRC_DIR = RUNTIME_DIR / "submission_src"
    SUB_DIR = RUNTIME_DIR / "submission"

    if not (SUB_SRC_DIR / "main.py").exists():
        logger.error(f"❌ 'main.py' introuvable dans {SUB_SRC_DIR}. Est-ce le bon dossier ?")
        sys.exit(1)

    try:
        cfg = OmegaConf.load(PROJECT_DIR / "0_configs/default.yaml")
    except Exception as e:
        logger.error(f"❌ Impossible de charger '0_configs/default.yaml': {e}")
        sys.exit(1)

    # ==========================================
    # 2. NETTOYAGE DES ANCIENS ARTEFACTS
    # ==========================================
    logger.info("🧹 Nettoyage des anciens artefacts...")
    to_remove = [
        SUB_SRC_DIR / "submission.csv",
        SUB_SRC_DIR / "src.zip",
        SUB_SRC_DIR / "weights.zip",
        SUB_DIR / "log.txt",
        SUB_DIR / "submission.csv",
        SUB_DIR / "submission.zip"
    ]
    for f in to_remove:
        if f.exists():
            f.unlink()
            
    if (SUB_SRC_DIR / "cache").exists():
        shutil.rmtree(SUB_SRC_DIR / "cache")
        
    for d in ["src", "weights"]:
        if (SUB_SRC_DIR / d).exists():
            shutil.rmtree(SUB_SRC_DIR / d)

    # ==========================================
    # 3. COPIE DU SOURCE LOCAL ET DES POIDS
    # ==========================================
    logger.info("🔄 Copie du code source local (src) et des poids...")
    
    dest_src = SUB_SRC_DIR / "src"
    shutil.copytree(PROJECT_DIR / "src", dest_src)
    logger.info(f"   -> Code source local copié depuis {PROJECT_DIR / 'src'}.")

    dest_weights = SUB_SRC_DIR / "weights"
    dest_weights.mkdir(parents=True, exist_ok=True)
    
    phase4_dir = Path(cfg.paths.phase4_out)
    if phase4_dir.exists():
        pth_files = list(phase4_dir.rglob("*.pth"))
        for pth in pth_files:
            shutil.copy2(pth, dest_weights / pth.name)
        logger.info(f"   -> {len(pth_files)} modèles Siamois copiés.")
    else:
        logger.warning(f"⚠️ Dossier Phase 4 introuvable : {phase4_dir}")

    phase2_dir = Path(cfg.paths.phase2_out)
    meta_model = phase2_dir / "meta_model_stacking.pkl"
    if meta_model.exists():
        shutil.copy2(meta_model, dest_weights / meta_model.name)
        logger.info("   -> Méta-modèle (Arbitre) copié.")

    # ==========================================
    # 4. ZIPPAGE TEMPORAIRE
    # ==========================================
    logger.info("📦 Création de src.zip et weights.zip...")
    run_cmd(["uvx", "rpzip", "-r", "src.zip", "src/"], cwd=SUB_SRC_DIR)
    run_cmd(["uvx", "rpzip", "-r", "weights.zip", "weights/"], cwd=SUB_SRC_DIR)

    logger.info("🗑️ Suppression temporaire des dossiers pour un packaging propre...")
    shutil.rmtree(dest_src)
    shutil.rmtree(dest_weights)

    # ==========================================
    # 5. PACKAGING DE LA SOUMISSION OFFICIELLE
    # ==========================================
    logger.info("⚙️ Lancement des commandes Just (Pack & Check)...")
    try:
        # Créera un submission.zip contenant UNIQUEMENT main.py, src.zip et weights.zip
        run_cmd(["just", "pack-submission"], cwd=RUNTIME_DIR)
        run_cmd(["just", "check-submission"], cwd=RUNTIME_DIR)
        
        # ==========================================
        # 6. RESTAURATION POUR L'INFÉRENCE LOCALE
        # ==========================================
        logger.info("📂 Restauration de src/ et weights/ pour exécution locale de main.py...")
        shutil.unpack_archive(SUB_SRC_DIR / "src.zip", SUB_SRC_DIR)
        shutil.unpack_archive(SUB_SRC_DIR / "weights.zip", SUB_SRC_DIR)
        
        # ==========================================
        # 7. TEST DOCKER
        # ==========================================
        logger.info("🐳 Lancement du test Docker (Inférence locale)...")
        run_cmd(["just", "test-submission"], cwd=RUNTIME_DIR)
        
        logger.success("🎉 SOUMISSION PACKAGÉE ET TESTÉE AVEC SUCCÈS !")
        logger.info(f"📁 Fichier final prêt pour soumission : {SUB_DIR / 'submission.zip'}")
        
    except subprocess.CalledProcessError:
        logger.error("❌ Le test Docker ou la création de la soumission a échoué.")
        sys.exit(1)

if __name__ == "__main__":
    main()