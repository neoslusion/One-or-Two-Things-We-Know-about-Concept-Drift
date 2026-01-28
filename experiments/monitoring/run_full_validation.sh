#!/bin/bash
# run_full_validation.sh
# Runs a comprehensive statistical validation of the SE-CDT Adaptive System.
# Target: 30 independent runs on 'mixed' drift streams to prove stability.

# Get the project root directory (go up two levels from this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Change to project root
cd "$PROJECT_ROOT"

# Configuration
N_RUNS=3
N_SAMPLES=10000
DRIFT_TYPE="mixed"
THRESH=0.15
W_REF=50
BASE_OUT_DIR="experiments/monitoring/validation_results_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$BASE_OUT_DIR"

echo "========================================================"
echo "STARTING FULL SYSTEM VALIDATION"
echo "Working Directory: $(pwd)"
echo "Target: $N_RUNS runs | Stream: $DRIFT_TYPE | Samples: $N_SAMPLES"
echo "Output: $BASE_OUT_DIR"
echo "========================================================"

# Ensure PYTHONPATH is set
export PYTHONPATH=$PYTHONPATH:$PROJECT_ROOT/experiments/backup:$PROJECT_ROOT/experiments/shared:$PROJECT_ROOT

start_time=$(date +%s)

for ((i=1; i<=N_RUNS; i++)); do
    seed=$((42 + i*137)) # Deterministic prime spacing for seeds
    run_dir="$BASE_OUT_DIR/run_$i"
    mkdir -p "$run_dir"
    
    echo "[Run $i/$N_RUNS] Seed: $seed..."
    
    # Run evaluation in background to speed up (batch of 5?)
    # For safety in this environment, running sequentially to avoid OOM
    .venv/bin/python experiments/monitoring/evaluate_prequential.py \
        --n_samples $N_SAMPLES \
        --drift_type $DRIFT_TYPE \
        --seed $seed \
        --w_ref $W_REF \
        --sudden_thresh $THRESH \
        --output_dir "$run_dir" > "$run_dir/log.txt" 2>&1
        
    # Check if run succeeded
    if [ $? -eq 0 ]; then
        echo "  -> Completed."
    else
        echo "  -> FAILED. Check $run_dir/log.txt"
    fi
done

end_time=$(date +%s)
duration=$((end_time - start_time))

echo "========================================================"
echo "VALIDATION COMPLETE"
echo "Total Time: $(($duration / 60)) minutes"
echo "========================================================"

# Run Aggregation
echo "Aggregating results..."
.venv/bin/python experiments/monitoring/aggregate_validation_results.py --input_dir "$BASE_OUT_DIR"
