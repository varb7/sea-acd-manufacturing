#!/bin/bash
# Simple bash script to submit all inference_*.sh files to SLURM using sbatch.tinygpu

SCRIPT_DIR="sea-reproduce"
LOG_DIR="logs"

# Create logs directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Find all inference scripts
INFERENCE_SCRIPTS=$(find "$SCRIPT_DIR" \
    -name "inference_*.sh" \
    ! -name "*prior.sh" \
    | sort)

if [ -z "$INFERENCE_SCRIPTS" ]; then
    echo "No inference_*.sh files found in $SCRIPT_DIR"
    exit 1
fi

# Count scripts
TOTAL=$(echo "$INFERENCE_SCRIPTS" | wc -l)
echo "Found $TOTAL inference scripts"
echo ""

# Submit each script
SUBMITTED=0
FAILED=0
JOB_IDS=()

for script in $INFERENCE_SCRIPTS; do
    SCRIPT_NAME=$(basename "$script")
    echo "============================================================"
    echo "Submitting: $SCRIPT_NAME"
    echo "============================================================"
    
    # Submit the script using sbatch.tinygpu
    if JOB_ID=$(sbatch.tinygpu "$script" 2>&1); then
        # Extract job ID from output (format: "Submitted batch job 12345")
        JOB_NUM=$(echo "$JOB_ID" | grep -oP '\d+' | head -1)
        if [ -n "$JOB_NUM" ]; then
            JOB_IDS+=("$JOB_NUM")
            echo "$SCRIPT_NAME: Submitted as job $JOB_NUM"
            ((SUBMITTED++))
        else
            echo "$SCRIPT_NAME: Submitted (job ID not captured)"
            ((SUBMITTED++))
        fi
    else
        echo "$SCRIPT_NAME: FAILED to submit"
        ((FAILED++))
    fi
    echo ""
done

# Print summary
echo "============================================================"
echo "SUMMARY"
echo "============================================================"
echo "Total scripts: $TOTAL"
echo "Successfully submitted: $SUBMITTED"
echo "Failed to submit: $FAILED"
if [ ${#JOB_IDS[@]} -gt 0 ]; then
    echo ""
    echo "Job IDs: ${JOB_IDS[*]}"
    echo ""
    echo "Check job status with: squeue -u \$USER"
    echo "Check job output in: logs/ directory (as specified in each script's #SBATCH --output)"
fi
echo ""

exit $((FAILED > 0 ? 1 : 0))

