#!/bin/bash
# Restart drift monitoring system with clean state

echo "════════════════════════════════════════════════════════════"
echo "Restarting Drift Monitoring System (Clean State)"
echo "════════════════════════════════════════════════════════════"

# Stop all processes
echo ""
echo "1. Stopping existing processes..."
./terminate_all.sh 2>/dev/null || true
docker compose down 2>/dev/null || true
sleep 2

# Clear old data
echo ""
echo "2. Clearing old detection data..."
rm -f shapedd_batches.csv
rm -f models/current_model.pkl
echo "   ✓ Removed shapedd_batches.csv (will be recreated)"
echo "   ✓ Removed current_model.pkl (will be recreated after drift)"

# Optional: Clear old snapshots (uncomment if needed)
# echo "   ✓ Clearing old snapshots..."
# rm -f snapshots/drift_window_*.npz

echo ""
echo "3. Rebuilding Docker adaptor (if Dockerfile changed)..."
docker compose build adaptor
echo "   ✓ Adaptor image rebuilt"

echo ""
echo "4. Starting services with fresh consumer group..."
echo "   (This will reset Kafka offset to start from beginning)"
RESET_OFFSET_ON_RESTART=true ./deploy.sh

echo ""
echo "════════════════════════════════════════════════════════════"
echo "System restarted with clean state!"
echo ""
echo "✓ CSV cleared (no old detections)"
echo "✓ Consumer will use fresh group ID (starts from earliest)"
echo "✓ Producer will start with new stream (idx resets to 0)"
echo ""
echo "Now you can run:"
echo "  python plot_detection_realtime.py"
echo ""
echo "The visualization will show only NEW data and detections!"
echo "════════════════════════════════════════════════════════════"

