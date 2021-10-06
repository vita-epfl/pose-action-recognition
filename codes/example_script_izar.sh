#!/bin/bash -l
# script from https://github.com/CS-433/cs-433-project-2-drop_table

#SBATCH --nodes=1
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu
#SBATCH --exclusive
#SBATCH --time 24:00:00
#SBATCH --mem=140G

module load gcc/8.4.0-cuda
module load python/3.7.7
module load mvapich2/2.3.4
module load py-torch/1.6.0-cuda-openmp
module load py-h5py/2.10.0-mpi
module load py-mpi4py/3.0.3
module load gcc python/3.7.7 cuda/10.2.89
pip install torch torchvision --user

srun --mem=140G python train.py