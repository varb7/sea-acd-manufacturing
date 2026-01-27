#!/bin/bash
# Submit only selected inference scripts via sbatch.tinygpu

SCRIPT_DIR="sea-reproduce"
LOG_DIR="logs"

mkdir -p "$LOG_DIR"

#  Put your chosen script filenames here (exact names)
SELECTED_SCRIPTS=(
  "inference_bossfci_ft.sh"
  "inference_rfci_ft.sh"
  "inference_gfci_ft.sh"
  "inference_cpc_ft.sh"
  "inference_fges_ft.sh"
  "inference_graspfci_ft.sh"

  
  
  # "inference_something_else.sh"
)

SUBMITTED=0
FAILED=0
JOB_IDS=()

for name in "${SELECTED_SCRIPTS[@]}"; do
  script="$SCRIPT_DIR/$name"

  if [ ! -f "$script" ]; then
    echo "MISSING: $name (expected at: $script)"
    ((FAILED++))
    continue
  fi

  echo "============================================================"
  echo "Submitting: $name"
  echo "============================================================"

  if JOB_OUT=$(sbatch.tinygpu "$script" 2>&1); then
    JOB_NUM=$(echo "$JOB_OUT" | grep -oP '\d+' | head -1)
    if [ -n "$JOB_NUM" ]; then
      JOB_IDS+=("$JOB_NUM")
      echo "$name: Submitted as job $JOB_NUM"
    else
      echo "$name: Submitted (job ID not captured)"
    fi
    ((SUBMITTED++))
  else
    echo "$name: FAILED to submit"
    echo "$JOB_OUT"
    ((FAILED++))
  fi

  echo ""
done

echo "============================================================"
echo "SUMMARY"
echo "============================================================"
echo "Requested scripts: ${#SELECTED_SCRIPTS[@]}"
echo "Successfully submitted: $SUBMITTED"
echo "Failed: $FAILED"
if [ ${#JOB_IDS[@]} -gt 0 ]; then
  echo "Job IDs: ${JOB_IDS[*]}"
  echo "Check status: squeue -u \$USER"
fi

exit $((FAILED > 0 ? 1 : 0))
