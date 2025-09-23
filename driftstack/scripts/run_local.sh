
#!/usr/bin/env bash
set -e
python -m venv .venv
source .venv/bin/activate || source .venv/Scripts/activate
pip install -r requirements.txt
python -m src.pipeline.run_stream --config configs/pipeline.yaml
