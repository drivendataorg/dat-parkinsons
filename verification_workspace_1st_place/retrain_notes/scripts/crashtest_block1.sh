#!/bin/bash
# Block 1 crash test: start every not-yet-run phase-1 training script at once, each capped at 3 minutes.
# Goal: "does it start and train without error", not result quality.
#   - runs from ~/sfmn1/runs/crashtest, whose 0_configs/default.yaml redirects phase1_out there and points
#     train_labels / folds_2d_csv at a stratified 100-scan subset (20 per fold); code and batch sizes unchanged
#   - MedicalNet variants read the existing ~/sfmn1/data/vol_cache (complete, 1362 volumes)
#   - OMP_NUM_THREADS=2 to limit CPU thread oversubscription (8 vCPU shared by 11 jobs)
# Usage: crashtest_block1.sh [job ...]   (no args = all 11 jobs)
C=/home/ec2-user/sfmn1/runs/crashtest
PY=/home/ec2-user/sfmn1/code/.venv/bin/python
T=/home/ec2-user/sfmn1/code/1_initial_training
CAP=${CAP:-180}
cd $C
export OMP_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

declare -A CMD=(
  [Vit2D]="$T/train_2d_vit.py"
  [DenseNet3D121]="$T/train_3d_densenet121.py"
  [DenseNet3D201]="$T/train_3d_densenet201.py"
  [DenseNetAttention]="$T/train_3d_attention.py"
  [SVM]="$T/train_svm.py"
  [MedicalNet18]="$T/train_medicalnet.py --run_name medicalnet18 --model medicalnet --medicalnet_backbone resnet --medicalnet_depth 18 --resnet_shortcut A"
  [MedicalNet34]="$T/train_medicalnet.py --run_name medicalnet34 --model medicalnet --medicalnet_backbone resnet --medicalnet_depth 34 --resnet_shortcut A"
  [MedicalNet200]="$T/train_medicalnet.py --run_name medicalnet200 --model medicalnet --medicalnet_backbone resnet --medicalnet_depth 200 --resnet_shortcut B"
  [MedicalNetResNext50]="$T/train_medicalnet.py --run_name medicalnet-resnext50 --model medicalnet --medicalnet_backbone resnext --medicalnet_depth 50 --resnet_shortcut B --resnext_cardinality 32"
  [MedicalNetResNext101]="$T/train_medicalnet.py --run_name medicalnet-resnext101 --model medicalnet --medicalnet_backbone resnext --medicalnet_depth 101 --resnet_shortcut B --resnext_cardinality 32"
  [MedNextB]="$T/train_medicalnet.py --run_name medicalnextB --model mednext"
)
JOBS=("$@"); [ ${#JOBS[@]} -eq 0 ] && JOBS=("${!CMD[@]}")
TAG=${TAG:-wave1}
mkdir -p logs

# GPU memory sampler (every 5 s) until all jobs finish
( while true; do echo "$(date +%T) $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader)"; sleep 5; done ) > logs/gpu_$TAG.log &
SAMPLER=$!

echo "start $(date +%T) cap ${CAP}s jobs: ${JOBS[*]}"
for j in "${JOBS[@]}"; do
  ( timeout -k 20 $CAP $PY -u ${CMD[$j]} > logs/${j}_$TAG.log 2>&1; echo "rc=$?" >> logs/${j}_$TAG.log ) &
done
wait $(jobs -p | grep -v "^$SAMPLER$") 2>/dev/null
# leftover DataLoader workers from killed jobs
pkill -f "$T/train_" 2>/dev/null
kill $SAMPLER
echo "end $(date +%T)"
for j in "${JOBS[@]}"; do echo "$j $(tail -1 logs/${j}_$TAG.log)"; done
