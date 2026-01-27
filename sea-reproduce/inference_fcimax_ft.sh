#!/bin/bash
#SBATCH --job-name=fcimaxft_inference
#SBATCH --output=logs/fcimax_%j.txt
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
python src/inference.py --config_file config/aggregator_tf_fcimax_manuft.yaml --run_name aggregator_tf_fcimaxft --checkpoint_path checkpoints/fcimax_manuft_nf/auprc=0.783_val_loss=0.000.ckpt --use_prior_knowledge

# Record the end time

end_time=$(date +%Y-%m-%d_%H:%M:%S)
echo "Job ended at: $end_time"
