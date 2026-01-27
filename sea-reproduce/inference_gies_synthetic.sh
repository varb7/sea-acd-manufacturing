#!/bin/bash
#SBATCH --job-name=gies_synthetic_inference
#SBATCH --output=logs/gies_synthetic_%j.txt
#SBATCH --nodes=1
#SBATCH --mail-user=varun.bhoj@fau.de
#SBATCH --ntasks-per-node=1
#SBATCH --time=15:30:00
#SBATCH --gres=gpu:1

# Record the start time
start_time=$(date +%Y-%m-%d_%H:%M:%S)
echo "Job started at: $start_time"

# Load any necessary modules
module load python
module load cudnn

# Navigate to project directory
cd /home/woody/iwfa/iwfa112h/sea-stable/sea-acd-manufacturing/sea-reproduce/

# Run GIES synthetic inference
python src/inference.py \
    --config_file config/aggregator_tf_gies.yaml \
    --run_name aggregator_tf_gies \
    --checkpoint_path checkpoints/gies_synthetic/model_best_epoch=535_auprc=0.849.ckpt

# Record the end time
end_time=$(date +%Y-%m-%d_%H:%M:%S)
echo "Job ended at: $end_time"
