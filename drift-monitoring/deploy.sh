#!/usr/bin/env bash
set -euo pipefail

# Default environment variables
export BROKERS=${BROKERS:-localhost:19092}
export TOPIC=${TOPIC:-sensor.stream}
export BUFFER_SIZE=${BUFFER_SIZE:-10000}
export CHUNK_SIZE=${CHUNK_SIZE:-250}
export DRIFT_PVALUE=${DRIFT_PVALUE:-0.05}
export RESULT_TOPIC=${RESULT_TOPIC:-drift.results}
export SHAPEDD_LOG=${SHAPEDD_LOG:-shapedd_batches.csv}

# Config for internal use
BROKERS_HOST="$BROKERS"

echo "[deploy] Environment:"
echo "  BROKERS=$BROKERS"
echo "  TOPIC=$TOPIC"
echo "  BUFFER_SIZE=$BUFFER_SIZE"
echo "  CHUNK_SIZE=$CHUNK_SIZE"
echo "  DRIFT_PVALUE=$DRIFT_PVALUE"
echo "  RESULT_TOPIC=$RESULT_TOPIC"
echo "  SHAPEDD_LOG=$SHAPEDD_LOG"

echo "[deploy] Restarting docker services..."
docker compose down 2>/dev/null || true
docker compose up -d

echo "[deploy] Waiting for Redpanda to be healthy..."
# Wait for admin API to respond
for i in {1..60}; do
  if curl -fsS http://localhost:9644/v1/status/ready >/dev/null 2>&1; then
    echo "[deploy] Redpanda is ready"
    break
  fi
  sleep 1
done

echo "[deploy] Ensuring topic '$TOPIC' exists (auto-create enabled)"

echo "[deploy] Launching shapedd consumer..."
BROKERS="$BROKERS_HOST" TOPIC="$TOPIC" nohup python3 consumer_stream.py > consumer.log 2>&1 &
CONSUMER_PID=$!
echo "[deploy] Consumer PID: $CONSUMER_PID"

sleep 2

echo "[deploy] Launching producer..."
BROKERS="$BROKERS_HOST" TOPIC="$TOPIC" nohup python3 producer.py > producer.log 2>&1 &
PRODUCER_PID=$!
echo "[deploy] Producer PID: $PRODUCER_PID"

echo "[deploy] Tail logs (Ctrl+C to stop). Logs also in producer.log / consumer.log"
trap 'echo; echo "[deploy] Stopping..."; kill $PRODUCER_PID $CONSUMER_PID 2>/dev/null || true; exit 0' INT
tail -f consumer.log producer.log


