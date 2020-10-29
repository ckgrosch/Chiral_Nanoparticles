#!/bin/bash
# Job name:
#SBATCH --job-name=ensemble_cnn_v2_20perror
#
# Account:
#SBATCH --account=fc_electron
#
# Partition:
#SBATCH --partition=savio2_gpu
#
# QoS:
#SBATCH --qos=savio_normal
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
#SBATCH --time=2:30:00
#
# Mail type:
#SBATCH --mail-type=all
#
# Mail user:
#SBATCH --mail-user=cgroschner@berkeley.edu
#
## Command(s) to run (example):
module load ml/tensorflow/1.12.0-py36
python cnn_chiral_ensemble_20perror_v2.py
