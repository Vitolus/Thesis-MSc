#!/bin/bash
#SBATCH --job-name=voice_generation
#SBATCH --nodes=1
#SBATCH --gres=gpu:tesla:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=/home/davide.vitagliano/thesis/voice_input.log
#SBATCH --chdir=/home/davide.vitagliano/thesis

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Error: You must provide input and output paths."
    echo "Usage: sbatch run_generation.slurm <input_csv> <output_dir>"
    exit 1
fi

echo "SLURM_NODELIST $SLURM_NODELIST"
echo "Processing Input: $1"
echo "Saving to Output: $2"

source /home/davide.vitagliano/miniforge3/etc/profile.d/conda.sh
conda activate pytorch
export PYTORCH_ALLOC_CONF=expandable_segments:True

srun python voice_input_generation_script.py -i "$1" -o "$2"

exit