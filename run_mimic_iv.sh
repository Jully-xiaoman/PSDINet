#!/usr/bin/env bash

CONFIG=./data/mimic-iv/config_m4.yaml


DATASETS=(
  "mimic-iv-ATC3"
  "mimic-iv-ATC4"
)
# 1) FFT 消融实验（衰减模块全开）
for ds in "${DATASETS[@]}"; do
   python ./src/train_MELORA.py \
      --config "${CONFIG}" \
      --dataset "${ds}" \
      --gpu 1
done

