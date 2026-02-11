import pandas as pd
import numpy as np
import os

def analyze_benchmarks():
    file_path = "experiments/benchmark_proper_results.pkl"
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    df = pd.read_pickle(file_path)
    print(f"Loaded {len(df)} experiment results.")
    
    # Expand SE_Classifications
    # Each row in df is a "Scenario" run, containing multiple events
    # We need to flatten this to get (GroundTruth, Prediction) pairs
    
    records = []
    for idx, row in df.iterrows():
        scenario = row.get("Scenario", "Unknown")
        se_results = row.get("SE_Classifications", [])
        
        for res in se_results:
            gt = res['gt_type'] # e.g. "Sudden", "Gradual"
            pred = res['pred'] # e.g. "Sudden", "Gradual", "Incremental"
            
            # Map GT to Category (TCD/PCD)
            if gt in ["Sudden", "Blip"]:
                gt_cat = "TCD"
            elif gt in ["Gradual", "Incremental", "Recurrent"]:
                gt_cat = "PCD"
            else:
                 gt_cat = "Unknown"
                 
            # Map Pred to Category
            if pred in ["Sudden", "Blip"]:
                pred_cat = "TCD"
            elif pred in ["Gradual", "Incremental", "Recurrent"]:
                pred_cat = "PCD"
            else:
                pred_cat = "Unknown"
                
            records.append({
                "Scenario": scenario,
                "GT_Type": gt,
                "GT_Cat": gt_cat,
                "Pred_Type": pred,
                "Pred_Cat": pred_cat
            })
            
    res_df = pd.DataFrame(records)
    
    if res_df.empty:
        print("No classification records found.")
        return

    # Calculate Accuracies by GT_Type
    print("\n=== SE-CDT Performance by Drift Type ===")
    print(f"{'Type':<15} {'Count':<5} {'CAT Acc':<10} {'SUB Acc':<10} {'Common Mistake':<20}")
    print("-" * 65)
    
    for dtype in ["Sudden", "Blip", "Gradual", "Incremental", "Recurrent"]:
        sub = res_df[res_df["GT_Type"] == dtype]
        if sub.empty:
            continue
            
        count = len(sub)
        cat_acc = (sub["GT_Cat"] == sub["Pred_Cat"]).mean() * 100
        sub_acc = (sub["GT_Type"] == sub["Pred_Type"]).mean() * 100
        
        # Most common error
        errors = sub[sub["GT_Type"] != sub["Pred_Type"]]
        if not errors.empty:
            common_mistake = errors["Pred_Type"].mode()[0]
        else:
            common_mistake = "None"
            
        print(f"{dtype:<15} {count:<5} {cat_acc:<10.1f} {sub_acc:<10.1f} {common_mistake:<20}")
        
    print("\n=== Detailed Confusion Matrix (Gradual) ===")
    grad_sub = res_df[res_df["GT_Type"] == "Gradual"]
    if not grad_sub.empty:
        print(grad_sub["Pred_Type"].value_counts())

    print("\n=== Detailed Confusion Matrix (Incremental) ===")
    inc_sub = res_df[res_df["GT_Type"] == "Incremental"]
    if not inc_sub.empty:
        print(inc_sub["Pred_Type"].value_counts())

if __name__ == "__main__":
    analyze_benchmarks()
