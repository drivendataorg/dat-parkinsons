import pandas as pd
from sklearn.model_selection import StratifiedKFold
from omegaconf import OmegaConf
from pathlib import Path


cfg = OmegaConf.merge(
    OmegaConf.load("0_configs/default.yaml"), 
    OmegaConf.from_cli()
)

df = pd.read_csv(f'{cfg.paths.root_dir}/train_labels_JNDlMjr.csv')

# On s'assure d'un ordre fixe avant le split
df = df.sort_values('uid').reset_index(drop=True)

df['fold'] = -1
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(df, df['is_pathologic'])):
    df.loc[val_idx, 'fold'] = fold_idx

# On sauvegarde et on partage ce fichier avec l'équipe (via Git, Drive, etc.)
df.to_csv(cfg.paths.train_labels, index=False)