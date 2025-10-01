#!/usr/bin/env bash
set -euo pipefail

# Config
BROKERS_HOST=${BROKERS:-localhost:19092}
TOPIC=${TOPIC:-sensor.stream}

echo "[deploy] Starting docker services..."
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


