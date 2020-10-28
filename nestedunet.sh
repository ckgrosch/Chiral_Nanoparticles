#!/bin/bash
# Job name:
#SBATCH --job-name=chiral_net_00perror_v2b
#
# Account:
#SBATCH --account=fc_electron
#
# Partition:
#SBATCH --partition=savio2_gpu
#
# QoS:
#SBATCH --qos=savio_debug
#
# Number of nodes:
#SBATCH --nodes=1
#
# Specify one task:
#SBATCH --ntasks-per-node=1
#
# Number of processors for single task needed for use case (example):
#SBATCH --cpus-per-task=2
#
#Number of GPUs, this can be in the format of "gpu:[1-4]", or "gpu:K80:[1-4] with the type included
#SBATCH --gres=gpu:1
#
# Wall clock limit:
#SBATCH --time=0:30:00
#
# Mail type:
#SBATCH --mail-type=all
#
# Mail user:
#SBATCH --mail-user=cgroschner@berkeley.edu
#
## Command(s) to run (example):
source env/bin/activate
module load cuda/9.0
module load pytorch/0.4.0-py36-cuda9.0
python train.py --dataset chiral_np --arch NestedUNet --img_ext .png --mask_ext .png --optimizer Adam --input_w 308 --input_h 308 --epochs 200
python val.py --name chiral_np_NestedUNet_woDS
