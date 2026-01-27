#!/bin/bash
#SBATCH --job-name=fcimax_prior_inference
#SBATCH --output=logs/fcimax_prior_%j.txt
#SBATCH --nodes=1
#SBATCH --mail-user=varun.bhoj@fau.de
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --partition=rtx3080
#SBATCH --gres=gpu:rtx3080:1

# Record the start time
start_time=$(date +%Y-%m-%d_%H:%M:%S)
echo "Job started at: $start_time"

# Load any necessary modules
module load python
module load cudnn

# Navigate to project directory
cd /home/woody/iwfa/iwfa112h/sea-stable/sea-acd-manufacturing/sea-reproduce/

# Run FCI inference
# NOTE: Add checkpoint path when available
python src/inference.py --config_file config/aggregator_tf_fcimax_prior.yaml --run_name aggregator_tf_fcimax_prior --checkpoint_path checkpoints/fci_synthetic/model_best_epoch=373_auprc=0.842.ckpt --use_prior_knowledge --edge_threshold 0.3 --temperature 2.0

# Record the end time
end_time=$(date +%Y-%m-%d_%H:%M:%S)
echo "Job ended at: $end_time"
