#!/usr/bin/env bash
# Usage: ./eval_dearli.sh <dataset> <split> <checkpoint_path>
# dataset: ade20k | coco-obj | coco-pan
# split:   ade20k -> 1_128 1_64 1_32 1_16 1_8
#          coco-* -> 1_512 1_256 1_128 1_64 1_32
# checkpoint_path: path to the model checkpoint to evaluate
# num gpus (optional): number of gpus to use (default: 1)

set -euo pipefail

[[ $# -eq 4 ]] || { echo "usage: $0 <dataset> <split> <checkpoint_path> <num_gpus>"; exit 1; }
DATASET="$1"; SPLIT="$2"; CHECKPOINT_WEIGHTS="$3"; NUM_GPUS="${4:-1}"

# Fixed config roots
ADE_CFG_DIR="./configs/ade20k/panoptic-segmentation-vlm/convnext_semisup"
COCO_OBJ_CFG_DIR="./configs/coco/panoptic-segmentation-vlm/convnext_semisup"
COCO_PAN_CFG_DIR="./configs/coco/panoptic-segmentation-vlm/convnext_semisup"

# Pick cfg dir
case "$DATASET" in
  ade20k)   CFG_DIR="$ADE_CFG_DIR" ;;
  coco-obj) CFG_DIR="$COCO_OBJ_CFG_DIR" ;;
  coco-pan) CFG_DIR="$COCO_PAN_CFG_DIR" ;;
  *) echo "Unknown dataset: $DATASET"; exit 1 ;;
esac

# Validate split
# Not important for validation, but keep for consistency
valid_ade=(1_128 1_64 1_32 1_16 1_8)
valid_coco=(1_512 1_256 1_128 1_64 1_32)
is_valid() { local x="$1"; shift; for v in "$@"; do [[ "$x" == "$v" ]] && return 0; done; return 1; }
if [[ "$DATASET" == "ade20k" ]]; then
  is_valid "$SPLIT" "${valid_ade[@]}" || { echo "bad split for ade20k: $SPLIT"; exit 1; }
else
  is_valid "$SPLIT" "${valid_coco[@]}" || { echo "bad split for $DATASET: $SPLIT"; exit 1; }
fi

# Config file (must exist)
CONFIG_FILE="$CFG_DIR/vlm_convnext_base_${DATASET}_${SPLIT}.yaml"
[[ -f "$CONFIG_FILE" ]] || { echo "config not found: $CONFIG_FILE"; exit 1; }

# Checkpoint must exist
[[ -f "$CHECKPOINT_WEIGHTS" ]] || { echo "checkpoint not found: $CHECKPOINT_WEIGHTS"; exit 1; }

METHOD_ARGS=(MODEL.WEIGHTS "$CHECKPOINT_WEIGHTS" )

# Show & run
echo "python train_net_semi_sup.py --eval-only --num-gpus $NUM_GPUS --config-file \"$CONFIG_FILE\" ${METHOD_ARGS[*]}"
python train_net_semi_sup.py --eval-only --num-gpus "$NUM_GPUS" --config-file "$CONFIG_FILE" "${METHOD_ARGS[@]}"