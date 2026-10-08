#!/usr/bin/env bash
# Train all 9 baselines (3 classes x 3 architectures) sequentially on one GPU.
#   GPU=0 DATA=data bash scripts/train_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."
GPU="${GPU:-0}"; DATA="${DATA:-data}"
mkdir -p logs
for cls in coronal_hole sunspot prominence; do
  for arch in unet deeplabv3 segformer; do
    echo "=== $cls / $arch ==="
    CUDA_VISIBLE_DEVICES="$GPU" python train.py --cls "$cls" --arch "$arch" --data-root "$DATA" \
      2>&1 | tee "logs/${cls}_${arch}.log"
  done
done
python scripts/summarize.py
