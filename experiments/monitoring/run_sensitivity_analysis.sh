#!/bin/bash
# run_sensitivity_analysis.sh
# Sensitivity Analysis for SE-CDT System

# Directory setup
mkdir -p experiments/drift_monitoring_system/sensitivity_results

# Parameters
DRIFT_TYPES=("sudden" "gradual" "recurrent")
THRESHOLDS=(0.3 0.5 0.7)
W_REFS=(30 50 75)

echo "Starting Sensitivity Analysis..."

# Ensure PYTHONPATH includes the backup directory for se_cdt
export PYTHONPATH=$PYTHONPATH:$(pwd)/experiments/backup

for dtype in "${DRIFT_TYPES[@]}"; do
    for thresh in "${THRESHOLDS[@]}"; do
        for w in "${W_REFS[@]}"; do
            echo "Running: Type=$dtype, Thresh=$thresh, W_Ref=$w"
            
            python experiments/drift_monitoring_system/evaluate_prequential.py \
                --drift_type "$dtype" \
                --sudden_thresh "$thresh" \
                --w_ref "$w" \
                --output_dir "experiments/drift_monitoring_system/sensitivity_results/${dtype}_t${thresh}_w${w}" \
                --n_samples 3000 \
                --n_drifts 3
                
        done
    done
done

echo "Sensitivity Analysis Complete."
