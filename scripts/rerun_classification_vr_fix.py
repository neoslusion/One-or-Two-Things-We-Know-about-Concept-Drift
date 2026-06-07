"""
Focused re-run of the SE-CDT *classification* benchmark only.

Replicates the classification loop of benchmark_proper.run_mixed_experiment
(lines ~387-422) for all 6 scenarios x 30 seeds, comparing:
  - BROKEN: classify(sig_slice, drift_length)            -> VR always 1.0
  - FIXED : classify(sig_slice, drift_length, data_window) -> VR computed

Concept-memory override is applied identically in both arms (it is independent
of the VR fix) so we isolate the effect of actually computing VR.
"""
import sys, collections
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "benchmark"))

from benchmark_proper import (
    generate_mixed_stream, compute_mmd_sequence, WINDOW_SIZE,
)
from core.detectors.se_cdt import SE_CDT

DRIFT_TYPES = ["Sudden", "Blip", "Gradual", "Incremental", "Recurrent"]
N_SEEDS = 30


def build_events(scenario):
    events = []
    if scenario == "Mixed_A":
        pattern = [{"type": "Sudden", "width": 0}, {"type": "Gradual", "width": 400},
                   {"type": "Recurrent", "width": 400}]
        for i in range(10):
            evt = pattern[i % 3]
            events.append({"type": evt["type"], "pos": 800 + i * 1000, "width": evt["width"]})
    elif scenario == "Mixed_B":
        pattern = [{"type": "Blip", "width": 100}, {"type": "Incremental", "width": 500},
                   {"type": "Sudden", "width": 0}]
        for i in range(10):
            evt = pattern[i % 3]
            events.append({"type": evt["type"], "pos": 800 + i * 1200, "width": evt["width"]})
    elif scenario == "Repeated_Sudden":
        events = [{"type": "Sudden", "pos": 800 + i * 800, "width": 0} for i in range(10)]
    elif scenario == "Repeated_Gradual":
        events = [{"type": "Gradual", "pos": 800 + i * 1000, "width": 1000} for i in range(10)]
    elif scenario == "Repeated_Recurrent":
        events = [{"type": "Recurrent", "pos": 800 + i * 1000, "width": 400} for i in range(10)]
    elif scenario == "Repeated_Incremental":
        events = [{"type": "Incremental", "pos": 800 + i * 1200, "width": 600} for i in range(10)]
    return events


def classify_stream(events, seed, pass_data_window):
    length = events[-1]['pos'] + 2000
    X_shared, _ = generate_mixed_stream(events, length, seed, supervised_mode=False)
    mmd_sig = compute_mmd_sequence(X_shared, WINDOW_SIZE, step=10, use_standard=True)
    se = SE_CDT(WINDOW_SIZE)

    out = []
    for evt in sorted(events, key=lambda e: e["pos"]):
        evt_pos = evt['pos']
        half_window = 750
        win_start = max(0, evt_pos - half_window)
        win_end = min(len(X_shared), evt_pos + half_window)
        data_window = X_shared[win_start:win_end]

        trace_idx = evt_pos // 10
        trace_half_window = max(10, half_window // 10)
        trace_start = max(0, trace_idx - trace_half_window)
        trace_end = min(len(mmd_sig), trace_idx + trace_half_window)
        sig_slice = mmd_sig[trace_start:trace_end]

        if len(data_window) >= 500 and len(sig_slice) >= 10:
            drift_length = se._growth_process(data_window, mmd_trace=sig_slice)
            if pass_data_window:
                res_se = se.classify(sig_slice, drift_length=drift_length, data_window=data_window)
            else:
                res_se = se.classify(sig_slice, drift_length=drift_length)

            if se.use_concept_memory:
                snapshot = se._extract_post_drift_snapshot(X_shared, evt_pos)
                if snapshot is not None:
                    recurrent_idx, _ = se._match_or_store_concept(snapshot)
                    if recurrent_idx >= 0 and res_se.subcategory != "Blip":
                        res_se.subcategory = "Recurrent"

            out.append({"gt_type": evt['type'], "pred": res_se.subcategory,
                        "VR": res_se.features.get("VR")})
    return out


def run(pass_data_window):
    conf = {dt: collections.Counter() for dt in DRIFT_TYPES}
    totals = collections.Counter()
    allpred = collections.Counter()
    vrs = collections.Counter()
    scenarios = ["Mixed_A", "Mixed_B", "Repeated_Sudden", "Repeated_Gradual",
                 "Repeated_Recurrent", "Repeated_Incremental"]
    for sc in scenarios:
        events = build_events(sc)
        for seed in range(N_SEEDS):
            for it in classify_stream(events, seed, pass_data_window):
                gt, pred = it["gt_type"], it["pred"]
                allpred[pred] += 1
                vrs[round(it["VR"], 2) if it["VR"] is not None else None] += 1
                if gt in DRIFT_TYPES:
                    totals[gt] += 1
                    conf[gt][pred] += 1
    return conf, totals, allpred, vrs


def report(tag, conf, totals, allpred, vrs):
    print(f"\n========== {tag} ==========")
    print("VR distribution:", dict(vrs))
    print("All predictions:", dict(allpred))
    accs = {}
    for dt in DRIFT_TYPES:
        n = totals[dt]
        acc = conf[dt][dt] / n * 100 if n else float('nan')
        accs[dt] = acc
        print(f"  {dt:12s} n={n:4d} acc={acc:5.1f}%  preds={dict(conf[dt])}")
    macro = np.mean([accs[dt] for dt in DRIFT_TYPES])
    correct = sum(conf[dt][dt] for dt in DRIFT_TYPES)
    tot = sum(totals[dt] for dt in DRIFT_TYPES)
    micro = correct / tot * 100 if tot else 0
    print(f"  Macro avg = {macro:.1f}%   Micro avg = {micro:.1f}%")


if __name__ == "__main__":
    for tag, flag in [("BROKEN (no data_window, VR=1.0)", False),
                      ("FIXED (data_window passed, VR computed)", True)]:
        conf, totals, allpred, vrs = run(flag)
        report(tag, conf, totals, allpred, vrs)
