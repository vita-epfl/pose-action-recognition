#!/bin/bash
#SBATCH --job-name RUN_TITAN
#SBATCH --account=vita
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --time 72:00:00
#SBATCH --output "out/slurm-%A_%a.log"
#SBATCH --mem=32G

echo started at `date`
module load gcc/8.4.0-cuda cuda/10.2.89
echo "${@:1}"
python -u "${@:1}"
wait 
echo finished at `date`