import os
import shutil
from pathlib import Path

# 1. Define your dictionary mapping model names to folder paths
model_paths = {
    # Medicalnet models
    "MedicalResnext50":             "/data/HOME/dvtle/dat-park/runs",
    "MedicalResnext101":            "/data/HOME/dvtle/dat-park/runs",
    "MedicalSeResnext50":           "/data/HOME/dvtle/dat-park/runs",
    "MedicalSeResnext101":          "/data/HOME/dvtle/dat-park/runs",
    "MedicalSeResnext50Card64":     "/data/HOME/dvtle/dat-park/runs",
    "MedicalSeResnext50Card128":    "/data/HOME/dvtle/dat-park/runs",
    "MedicalAttnResnext101":        "/data/HOME/dvtle/dat-park/runs",
    "MedicalNet10":                 "/data/HOME/dvtle/dat-park/runs",
    "MedicalNet18":                 "/data/HOME/dvtle/dat-park/runs",
    "MedicalNet34":                 "/data/HOME/dvtle/dat-park/runs",
    "MedicalNet50":                 "/data/HOME/dvtle/dat-park/runs",
    "MedicalNet200":                "/data/HOME/dvtle/dat-park/runs",
    "MedNextB":                     "/data/HOME/dvtle/dat-park/runs",
    "MedNextM":                     "/data/HOME/dvtle/dat-park/runs",
    # 2DCNN + RNN
    "SeResNext26DGRU":              "/data/HOME/dvtle/dat-park/runs_2d_rnn",
    "Resnet18DLSTM":                "/data/HOME/dvtle/dat-park/runs_2d_rnn",
    "Resnet18DGRU":                 "/data/HOME/dvtle/dat-park/runs_2d_rnn",
    "Resnet34DGRU":                 "/data/HOME/dvtle/dat-park/runs_2d_rnn",
    # 2DCNN + RNN + Coronal & Axial slices
    "SeResNext26DLSTMCorax":        "/data/HOME/dvtle/dat-park/runs_2d_corax",
    "SeResNext26TSGRUCorax":        "/data/HOME/dvtle/dat-park/runs_2d_corax",
    "SeResNext26DGRUCorax":         "/data/HOME/dvtle/dat-park/runs_2d_corax",
    # 2DCNN + RNN + Sagittal + Coronal & Axial slices
    "SeResNext26DGRUSagcorax":       "/data/HOME/dvtle/dat-park/runs_2d_sagcorax",
    "SeResNext26DLSTMSagcorax":      "/data/HOME/dvtle/dat-park/runs_2d_sagcorax",
    "SeResNext26TSLSTMSagcorax":     "/data/HOME/dvtle/dat-park/runs_2d_sagcorax",
    # 2DCNN_channels
    "Resnet18D2DChannel":           "/data/HOME/dvtle/dat-park/runs_2d",
    "Resnet34D2DChannel":           "/data/HOME/dvtle/dat-park/runs_2d",
    "SeResNext26D2DChannel":        "/data/HOME/dvtle/dat-park/runs_2d",
    "SeResNext26TS2DChannel":       "/data/HOME/dvtle/dat-park/runs_2d",
    "SeResNext26T2DChannel":        "/data/HOME/dvtle/dat-park/runs_2d",
    "RegnetY0022DChannel":          "/data/HOME/dvtle/dat-park/runs_2d",
    "RegnetY0042DChannel":          "/data/HOME/dvtle/dat-park/runs_2d",
    # 3D models Samuel
    "DenseNet3D121":                "/data/HOME/dvtle/dat-park/runs_2d/samuel",
    "DenseNet3D201":                "/data/HOME/dvtle/dat-park/runs_2d/samuel",
    "ResNet2D":                     "/data/HOME/dvtle/dat-park/runs_2d/samuel",
    "Vit2D":                        "/data/HOME/dvtle/dat-park/runs_2d/samuel",
    "DenseNetAttention":            "/data/HOME/dvtle/dat-park/runs_2d/samuel",
    "Siamese":                      "/data/HOME/sansel/dat-park/runs_siamois",
    "Siamese2.5d":                  "/data/HOME/sansel/dat-park/runs_siamois",
    "Siamese_Mashup":               "/data/HOME/sansel/dat-park/runs_siamois",
    "Siamese_Mashup_Align":         "/data/HOME/sansel/dat-park/runs_siamois",
    "Siamese_Synth":                "/data/HOME/sansel/dat-park/runs_siamois",
    "Siamese_Mashup_DTW_Linear":    "/data/HOME/dvtle/dat-park/runs_siamois",
    "Siamese_Mashup_DTW_Attn":      "/data/HOME/dvtle/dat-park/runs_siamois",
}

# >>>>>>>>>>>>>>>>>>>>>>>> CUSTOM <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
from constants import prefix_map
PATH_META = '/data/HOME/dvtle/dat-park/runs/stacking_v9/meta_model_stacking.pkl'
# >>>>>>>>>>>>>>>>>>>>>>>> END CUSTOM <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

def copy_weights(
    model_paths: dict, prefix_map: dict, output_dir: str = "weights"
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Loop over prefix_map instead of model_paths
    for model_name, prefix in prefix_map.items():
        # Check if the model exists in model_paths
        if model_name not in model_paths:
            print(f"[SKIP] Key '{model_name}' not found in model_paths.")
            continue

        source_folder = model_paths[model_name]

        # Clean prefix for folder naming if needed
        if prefix.startswith("2drnn_"):
            prefix = prefix.replace("2drnn_", "")
        elif prefix.startswith("2d_"):
            prefix = prefix.replace("2d_", "")
        folder_name = prefix

        src_dir = Path(source_folder) / folder_name

        if not src_dir.exists():
            print(
                f"[WARNING] Directory does not exist for '{model_name}': {src_dir}"
            )
            continue

        # Search for files starting with the specified prefix
        matching_files = [
            f
            for f in src_dir.iterdir()
            if f.is_file() and f.name.startswith(prefix)
        ]

        if not matching_files:
            print(
                f"[INFO] No files matching prefix '{prefix}' found in {src_dir}"
            )
            continue

        # Copy matching files to the target folder
        for file in matching_files:
            dest_file = output_path / file.name
            shutil.copy2(file, dest_file)
            print(f"[COPIED] {file.name} -> {output_path}/")

    # Copy metadata model file
    meta_path = Path(PATH_META)
    if meta_path.exists():
        shutil.copy2(meta_path, output_path)
        print(f"[COPIED] {PATH_META} -> {output_path}/")
    else:
        print(f"[WARNING] PATH_META file does not exist: {PATH_META}")


# Run the function
if __name__ == "__main__":
    copy_weights(model_paths, prefix_map)