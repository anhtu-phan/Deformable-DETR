#!/bin/bash

NAME="deformable"
PROJECT_NAME="protector"
NUM_CLASSES=8
coco_path=/usr/src/datasets/$PROJECT_NAME

cd ./models/ops
sh ./make.sh
cd ../..

python -u main.py --output_dir ./logs/$PROJECT_NAME/$NAME --coco_path $coco_path --epochs 100 --resume ./exps/r50_deformable_detr-checkpoint.pth --wandb_name $NAME --wandb_project_name $PROJECT_NAME --num_classes $NUM_CLASSES --finetune_ignore label_enc.weight class_embed