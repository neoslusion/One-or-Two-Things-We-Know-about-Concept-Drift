
# DriftStack Template (Concept Drift Detection & Adaptive Serving)

A minimal-yet-practical template to build a **concept drift detection** and **adaptive model** pipeline (Python-first).
It captures the stack you requested: detectors (ADWIN, D3, ShapeDD stub), online learners (River ARF), a policy engine,
and a runnable local pipeline that simulates a data stream, detects drift, adapts the model, and logs metrics.

> **Quick start**
> ```bash
> # 1) Create venv & install deps
> python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
> pip install -r requirements.txt
>
> # 2) Run the local streaming demo (synthetic data)
> python -m src.pipeline.run_stream --config configs/pipeline.yaml
> ```
>
> Optional:
> - `notebooks/evaluation.ipynb`: Start with prequential metrics & MTTD calculation.
> - `docker/Dockerfile`: Containerize the pipeline.
> - `docker/docker-compose.yml`: (commented services) placeholders for Kafka/MinIO/MLflow.

## Repo layout
```
src/
  detectors/         # ADWIN, D3, ShapeDD(stub)
  learners/          # River Adaptive Random Forest wrapper
  policy/            # Policy engine: map drift->action
  pipeline/          # Entry-point + stream runner
  utils/             # Logging, metrics
data/                # Synthetic generator(s)
configs/             # YAML configs
notebooks/           # Evaluation notebook skeleton
docker/              # Dockerfile, optional compose
scripts/             # Convenience scripts
```
## What’s included
- **Detectors**: ADWIN (river), D3 (AUC split), ShapeDD (placeholder, plug your impl).
- **Learners**: River Adaptive Random Forest (ARF) with partial_fit for online learning.
- **Policy engine**: baseline actions for abrupt vs gradual drift (reset vs decay) with cooldown/hysteresis.
- **Metrics**: prequential accuracy, MTTD, TP/FP/FN, runtime/instance.
- **Config-driven**: tweak windows, thresholds, learner params via YAML.

## Next steps
- Swap `data/synthetic.py` with your stream or connect Kafka/Flink later.
- Replace `shapedd_detector.py` with your ShapeDD implementation.
- Extend policy rules with recurring-concept cache + active labeling.
- Wire MLflow/Evidently (hooks added in code).
