CONFIG=./data/mimic-iii/config_m3.yaml

#DATASETS=(
#  "mimic-iii-ATC3"
#  "mimic-iii-ATC4"
#)

DATASETS=(
  "mimic-iii-ATC3"
)
for ds in "${DATASETS[@]}"; do
    python ./src/train_MELORA.py \
          --config "${CONFIG}" \
          --dataset "${ds}" \
          --gpu 1
done
