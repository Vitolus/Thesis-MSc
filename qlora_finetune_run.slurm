#!/bin/bash
#SBATCH --job-name=QLoRA_finetune
#SBATCH --nodes=1
#SBATCH --gres=gpu:tesla:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --output=/home/davide.vitagliano/thesis/qlora_%j.log
#SBATCH --chdir=/home/davide.vitagliano/thesis

echo "Starting QLoRA Finetuning Pipeline on Slurm..."
echo "SLURM_NODELIST $SLURM_NODELIST"

MASTER_PORT=$(shuf -i 29501-29700 -n 1)
NUM_GPUS=$SLURM_GPUS_ON_NODE

source /home/davide.vitagliano/miniforge3/etc/profile.d/conda.sh
conda activate pyqwen
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HF_HUB_ENABLE_HF_TRANSFER=1

torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT qlora_finetune_script.py

echo "QLoRA Finetuning Complete."