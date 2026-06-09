#!/bin/bash
#SBATCH --job-name=glados_distill
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6 # Match this to n_threads in the Python script
#SBATCH --gres=gpu:tesla:1
#SBATCH --array=0-3
#SBATCH --output=/home/davide.vitagliano/thesis/target_dist_train_%a.log
#SBATCH --chdir=/home/davide.vitagliano/thesis

echo "Starting Distillation Pipeline on Slurm..."
echo "SLURM_NODELIST $SLURM_NODELIST"

source /home/davide.vitagliano/miniforge3/etc/profile.d/conda.sh
conda activate pyqwen
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HF_HUB_ENABLE_HF_TRANSFER=1

python target_distillation_script.py \
    --input_csv "./data/description_prompts_train.csv" \
    --output_csv "./data/multimodal_ground_truth_train.csv" \
    --hf_repo "unsloth/Qwen3-14B-GGUF" \
    --hf_filename "Qwen3-14B-UD-Q4_K_XL.gguf" \
    --limit 0 \
    --array_id $SLURM_ARRAY_TASK_ID \
    --array_tasks $SLURM_ARRAY_TASK_COUNT

echo "Distillation Complete."