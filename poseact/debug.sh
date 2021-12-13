#!/bin/bash
#SBATCH --job-name debug
#SBATCH --account=vita
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --time 1:00:00
#SBATCH --output "out/slurm-%A_%a.log"
#SBATCH --mem=64G

echo started at `date`
module load gcc/8.4.0-cuda cuda/10.2.89

source /home/wexiong/anaconda3/bin/activate base 
conda activate pytorch 

echo "${@:1}"
python -u "${@:1}"
wait 
echo finished at `date`