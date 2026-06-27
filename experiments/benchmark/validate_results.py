import pandas as pd
import numpy as np
import os

RESULTS_FILE = "experiments/benchmark_proper_results.pkl"

def validate_cdt_msw(variance):
    # Paper criteria: Variance > 0.0005 is PCD, else TCD
    threshold = 0.0005
    return "PCD" if variance > threshold else "TCD"

def validate_se_cdt(features):
    # Paper criteria:
    # Sudden: WR < 1.2, SNR > 4
    # Gradual: WR > 2.0
    # Recurrent: CV < 0.2
    
    wr = features.get("WR", 0)
    snr = features.get("SNR", 0)
    cv = features.get("CV", 1.0)
    
    matches_sudden = (wr < 1.2) and (snr > 4)
    matches_gradual = (wr > 2.0)
    matches_recurrent = (cv < 0.2)
    
    return {
        "Matches_Sudden_Criteria": matches_sudden,
        "Matches_Gradual_Criteria": matches_gradual,
        "Matches_Recurrent_Criteria": matches_recurrent,
        "WR": wr, "SNR": snr, "CV": cv
    }

def run_validation():
    if not os.path.exists(RESULTS_FILE):
        print(f"File not found: {RESULTS_FILE}")
        return

    df = pd.read_pickle(RESULTS_FILE)
    print(f"Loaded {len(df)} results from {RESULTS_FILE}")
    
    validation_records = []
    
    for idx, row in df.iterrows():
        dtype = row["Drift Type"]
        
        # 1. CDT Validation
        # Check if the internal variance matches the TCD/PCD logic
        # Note: CDT_Prediction in df might be 'Blip', 'Sudden' etc.
        # We want to check if the VARIANCE aligns with the paper's claimed threshold.
        cdt_var = row["CDT_Variance"]
        cdt_decision_paper = validate_cdt_msw(cdt_var)
        
        # 2. SE Validation
        se_feats = row["SE_Features"]
        if not se_feats: se_feats = {}
        se_criteria = validate_se_cdt(se_feats)
        
        rec = {
            "Drift Type": dtype,
            "CDT_Var": cdt_var,
            "CDT_Paper_Decision": cdt_decision_paper,
            "SE_Paper_Sudden": se_criteria["Matches_Sudden_Criteria"],
            "SE_Paper_Gradual": se_criteria["Matches_Gradual_Criteria"],
            "SE_Paper_Recurrent": se_criteria["Matches_Recurrent_Criteria"],
            "WR": se_criteria["WR"],
            "CV": se_criteria["CV"]
        }
        validation_records.append(rec)
        
    val_df = pd.DataFrame(validation_records)
    
    # Summary Report
    print("\n=== VALIDATION AGAINST PAPER CRITERIA ===")
    
    # Check 1: Does CDT Variance accurately split TCD/PCD?
    # TCD Types: Sudden, Blip
    # PCD Types: Gradual, Incremental, Recurrent
    tcd_types = ["Sudden", "Blip"]
    pcd_types = ["Gradual", "Incremental", "Recurrent"]
    
    tcd_correct = val_df[val_df["Drift Type"].isin(tcd_types)]["CDT_Paper_Decision"].value_counts(normalize=True).get("TCD", 0)
    pcd_correct = val_df[val_df["Drift Type"].isin(pcd_types)]["CDT_Paper_Decision"].value_counts(normalize=True).get("PCD", 0)
    
    print(f"CDT_MSW TCD Validation (Sudden/Blip): {tcd_correct*100:.1f}% decided as TCD (Threshold 0.0005)")
    print(f"CDT_MSW PCD Validation (Grad/Inc/Rec): {pcd_correct*100:.1f}% decided as PCD (Threshold 0.0005)")
    
    # Check 2: SE Criteria Matches
    sudden_match = val_df[val_df["Drift Type"] == "Sudden"]["SE_Paper_Sudden"].mean()
    gradual_match = val_df[val_df["Drift Type"] == "Gradual"]["SE_Paper_Gradual"].mean()
    recurrent_match = val_df[val_df["Drift Type"] == "Recurrent"]["SE_Paper_Recurrent"].mean()
    
    print(f"\nSE_CDT Criteria Validation:")
    print(f"Sudden Cases meeting (WR < 1.2, SNR > 4): {sudden_match*100:.1f}%")
    print(f"Gradual Cases meeting (WR > 2.0): {gradual_match*100:.1f}%")
    print(f"Recurrent Cases meeting (CV < 0.2): {recurrent_match*100:.1f}%")
    
    # Detailed Failure Analysis (optional)
    # ... (Previous code)
    
    # Check 3: Runtime and Latency (MTTD)
    # Filter for single-event experiments (Standard Benchmark) where GT Position = 1000
    # Mixed experiments have 'Scenario' key, standard ones don't (or it's NaN/empty)
    
    std_df = df[df["Scenario"].isna() | (df["Scenario"] == "")] if "Scenario" in df.columns else df
    
    print("\n=== RUNTIME & LATENCY METRICS ===")
    
    # Runtime
    avg_cdt_time = std_df["CDT_Time"].mean()
    avg_se_time = std_df["SE_Time"].mean()
    print(f"Avg CDT_MSW Runtime: {avg_cdt_time:.4f} s")
    print(f"Avg SE_CDT Runtime:  {avg_se_time:.4f} s")
    
    # MTTD (Mean Time To Detection)
    # GT Position = 1000 samples. Window Size = 50.
    # Drift Position is in WINDOWS. 
    # Detected Sample = pos * 50.
    # Delay = (pos * 50) - 1000.
    
    # Only for True Positives (drift_detected)
    detected_df = std_df[std_df["CDT_Position"] != -1].copy()
    if not detected_df.empty:
        detected_df["Detection_Sample"] = detected_df["CDT_Position"] * 50
        detected_df["Delay"] = detected_df["Detection_Sample"] - 1000
        
        # Breakdown by Drift Type
        print("\nMTTD (samples) by Drift Type:")
        mttd_stats = detected_df.groupby("Drift Type")["Delay"].agg(["mean", "std", "count"])
        print(mttd_stats)
    else:
        print("No drifts detected for MTTD calculation.")
        
if __name__ == "__main__":
    run_validation()
