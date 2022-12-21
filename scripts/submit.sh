#!/bin/bash
#
#SBATCH --job-name=deformable
#SBATCH --output=deformable.txt
#SBATCH --ntasks=1
#SBATCH --partition=students
#SBATCH --gres=gpu:mem11g:1
#SBATCH --nodelist=gpu09
#SBATCH --mem=16000
#SBATCH --mail-user=anhtu@cl.uni-heidelberg.de
#SBATCH --mail-type=ALL

source ./ven/bin/activate
srun ./scripts/run.sh