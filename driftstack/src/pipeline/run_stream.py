
import argparse, time, os, json
import numpy as np
import yaml
from tqdm import tqdm

from src.utils.logging_config import setup_logging
from src.utils.metrics import Metrics, DriftEvent
from src.policy.policy_engine import PolicyEngine, PolicyConfig
from src.learners.river_arf import OnlineARF

from src.detectors.adwin_detector import ADWINDetector
from src.detectors.d3_detector import D3Detector
# from src.detectors.shapedd_detector import ShapeDDDetector

from data.synthetic import make_stream

def to_xdict(x_row):
    return {f"f{i}": float(v) for i, v in enumerate(x_row)}

def build_detectors(cfg):
    dets = []
    for d in cfg.get('detectors', []):
        if d['type'].upper() == 'ADWIN':
            dets.append(ADWINDetector(**d.get('params', {})))
        elif d['type'].upper() == 'D3':
            dets.append(D3Detector(**d.get('params', {})))
        # elif d['type'].upper() == 'SHAPEDD':
        #     dets.append(ShapeDDDetector(**d.get('params', {})))
    return dets

def main(args):
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    out_dir = cfg['logging']['out_dir']
    os.makedirs(out_dir, exist_ok=True)
    logger = setup_logging(out_dir)

    # Data
    X, y, true_drifts = make_stream(**cfg['stream'])
    logger.info(f"Generated stream: X={X.shape}, #true_drifts={len(true_drifts)}\n")

    # Metrics
    m = Metrics(true_drifts=true_drifts)

    # Detectors & policy
    detectors = build_detectors(cfg)
    pol = PolicyEngine(PolicyConfig(**cfg['policy']))

    # Learner
    learner = OnlineARF(**cfg['learner']['params'])

    # Loop
    last_action = None
    for t in tqdm(range(len(X))):
        x_t = X[t]; y_t = int(y[t])
        x_dict = to_xdict(x_t)

        # Predict -> update metrics -> learn
        y_pred = learner.learn_predict(x_dict, y_t)
        m.update_preq(y_t, y_pred)

        # Error stream for ADWIN (1 if wrong, else 0)
        err = 0 if y_pred == y_t else 1

        # Update detectors
        for det in detectors:
            if det.__class__.__name__.startswith("ADWIN"):
                score = det.update(err)
            else:
                score = det.update(x_t)
            if score >= 1.0:
                # classify drift kind (simple heuristic: if detector is D3 -> structural/abrupt-ish)
                kind = 'abrupt' if det.__class__.__name__.startswith("D3") else 'gradual'
                m.detections.append(DriftEvent(t=t, detector=det.name, score=float(score), kind=kind))
                action = pol.decide(t, det.name, score, drift_kind=kind)
                if action:
                    # Apply adaptation (simple policies)
                    if action == 'reset':
                        # re-instantiate learner
                        learner = OnlineARF(**cfg['learner']['params'])
                        last_action = ('reset', t, det.name)
                        logger.info(f"[{t}] Drift by {det.name}. Action: RESET model.\n")
                    elif action == 'decay':
                        # Placeholder: in River ARF we can simulate by temporarily reducing lambda_value or partial resets
                        last_action = ('decay', t, det.name)
                        logger.info(f"[{t}] Drift by {det.name}. Action: DECAY (placeholder).\n")
                    elif action == 'recall':
                        # Placeholder for recalling cached model
                        last_action = ('recall', t, det.name)
                        logger.info(f"[{t}] Drift by {det.name}. Action: RECALL (placeholder).\n")

        if (t+1) % cfg['logging']['report_every'] == 0:
            TP, FP, FN = m.confusion_counts()
            logger.info(f"[{t}] preq_acc={m.preq_accuracy:.4f} TP={TP} FP={FP} FN={FN} last_action={last_action}\n")

    # Final report
    TP, FP, FN = m.confusion_counts()
    report = {
        "prequential_accuracy": m.preq_accuracy,
        "MTTD": m.mttd(),
        "TP": TP, "FP": FP, "FN": FN,
        "n_true_drifts": len(true_drifts),
        "n_detections": len(m.detections)
    }
    artifacts_dir = cfg['logging']['artifacts_dir']
    os.makedirs(artifacts_dir, exist_ok=True)
    with open(os.path.join(artifacts_dir, "final_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print("\n=== FINAL REPORT ===")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/pipeline.yaml")
    args = p.parse_args()
    main(args)
