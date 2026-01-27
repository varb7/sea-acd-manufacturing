#!/bin/bash
#SBATCH --job-name=baseline_inference
#SBATCH --output=logs/baseline_%j.txt
#SBATCH --nodes=1
#SBATCH --mail-user=varun.bhoj@fau.de
#SBATCH --ntasks-per-node=1
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:1

# Record the start time
start_time=$(date +%Y-%m-%d_%H:%M:%S)
echo "Job started at: $start_time"

# Load any necessary modules
module load python
module load cudnn

# Navigate to project directory
cd /home/woody/iwfa/iwfa112h/sea-stable/sea-acd-manufacturing/sea-reproduce/

# Run Baseline inference (no checkpoint needed)
python src/inference.py \
    --config_file config/baseline.yaml \
    --run_name baseline

# Record the end time
end_time=$(date +%Y-%m-%d_%H:%M:%S)
echo "Job ended at: $end_time"
