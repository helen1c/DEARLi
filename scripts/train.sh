#!/usr/bin/env bash
# Usage: ./train_dearli.sh <dataset> <split> <method>
# dataset: ade20k | coco-obj | coco-pan
# split:   ade20k -> 1_128 1_64 1_32 1_16 1_8
#          coco-* -> 1_512 1_256 1_128 1_64 1_32
# method:  dear | dearli

set -euo pipefail

[[ $# -eq 3 ]] || { echo "usage: $0 <dataset> <split> <method>"; exit 1; }
DATASET="$1"; SPLIT="$2"; METHOD="$3"

# Fixed config roots
ADE_CFG_DIR="./configs/ade20k/panoptic-segmentation-vlm/convnext_semisup"
COCO_OBJ_CFG_DIR="./configs/coco/panoptic-segmentation-vlm/convnext_semisup"
COCO_PAN_CFG_DIR="./configs/coco/panoptic-segmentation-vlm/convnext_semisup"

# if method is dearli, assure you downloaded decoder warmup weights folder from:
# https://ferhr-my.sharepoint.com/:f:/g/personal/imartinovic_fer_hr/EkRyHVha3khAlDyimjY2sZUBvn5G5pVh5IONpG2I0ddy6A?e=a1825d
# and put that folder into ./checkpoints

DEARLI_APPENDIX=""
# if dataset is ade20k then ade20k, if coco-obj then "coco", if coco-pan then "coco"
case "$DATASET" in
  ade20k)   DEARLI_APPENDIX="ade20k" ;;
  coco-obj) DEARLI_APPENDIX="coco" ;;
  coco-pan) DEARLI_APPENDIX="coco" ;;
esac
DEARLI_WEIGHTS=./checkpoints/decoder_warmup_weights/decoder_warmup_on_$DEARLI_APPENDIX.pth

# Ensure weights file exists
if [[ "$METHOD" == "dearli" && ! -f "$DEARLI_WEIGHTS" ]]; then
  echo "DEARLi weights not found: $DEARLI_WEIGHTS"
  exit 1
fi

# Pick cfg dir
case "$DATASET" in
  ade20k)   CFG_DIR="$ADE_CFG_DIR" ;;
  coco-obj) CFG_DIR="$COCO_OBJ_CFG_DIR" ;;
  coco-pan) CFG_DIR="$COCO_PAN_CFG_DIR" ;;
  *) echo "Unknown dataset: $DATASET"; exit 1 ;;
esac

# Validate split
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

METHOD_ARGS=()
[[ "$METHOD" == "dear" || "$METHOD" == "dearli" ]] || { echo "unknown method: $METHOD"; exit 1; }
if [[ "$METHOD" == "dearli" ]]; then
  METHOD_ARGS+=(MODEL.WEIGHTS ${DEARLI_WEIGHTS} MODEL.TEACHER.WEIGHTS ${DEARLI_WEIGHTS})
fi

# append method args
NUM_GPUS=1
# actually 2x8, 8 labeled and 8 unlabeled
BATCH_SIZE=8
SEED=1
METHOD_ARGS+=(SOLVER.IMS_PER_BATCH $BATCH_SIZE SEED $SEED)

# first echo full command
echo "python train_net_semi_sup.py --config-file \"$CONFIG_FILE\" ${METHOD_ARGS[*]}"

# Train
python train_net_semi_sup.py --num-gpus "$NUM_GPUS" --config-file "$CONFIG_FILE" "${METHOD_ARGS[@]}"
