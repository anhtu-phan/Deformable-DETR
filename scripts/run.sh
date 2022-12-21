#!/bin/bash

NAME="dab_detr"
PROJECT_NAME="visdrone"
NUM_CLASSES=3

python -u main.py --output_dir ./logs/$PROJECT_NAME/$NAME --coco_path ~/detr/datasets/$PROJECT_NAME --wandb_name deformable --lr 2e-5 --epochs 150 --resume ./exps/r50_deformable_detr-checkpoint.pth --wandb_name $NAME --wandb_project_name $PROJECT_NAME --num_classes $NUM_CLASSES