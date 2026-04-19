"""Verify ALL numbers cited from CDT-MSW paper (Guo et al. 2022).

All values manually transcribed from Tables 4, 5, 6, 7 in the paper PDF.
Each cell value is from the paper's printed tables.

Run this script to verify the numbers used in our thesis are correctly
extracted from the paper.
"""
import numpy as np

print("=" * 80)
print("VERIFY CDT-MSW NUMBERS FROM PAPER (Guo et al. 2022)")
print("=" * 80)

print("\n--- Table 6: ACCcat per dataset x window size (CDT_MSW) ---")
table6 = {
    "Banana":        [0.8462, 0.7692, 0.6923, 0.8182, 0.7857],
    "Titanic":       [0.9286, 1.0000, 1.0000, 1.0000, 0.9333],
    "Thyroid":       [1.0000, 0.9333, 1.0000, 0.7500, 0.7692],
    "Diabetis":      [0.9286, 1.0000, 0.8000, 1.0000, 0.8571],
    "Breast_cancer": [0.8667, 1.0000, 0.9286, 1.0000, 1.0000],
    "Heart":         [0.8182, 1.0000, 1.0000, 1.0000, 1.0000],
    "German":        [0.8667, 0.9333, 1.0000, 1.0000, 0.9286],
    "KDDCup99":      [1.0000, 0.8000, 0.8182, 0.6667, 0.6000],
    "MITface":       [0.5000, 0.8333, 0.6364, 1.0000, 0.6000],
    "USPS":          [0.6667, 0.7778, 0.8000, 0.8571],
}
all_vals = [v for vals in table6.values() for v in vals]
cat_micro = np.mean(all_vals)
cat_macro = np.mean([np.mean(vs) for vs in table6.values()])
print(f"  Total cells = {len(all_vals)} (expected 49 = 9*5 + 4 since USPS s=20 is missing)")
print(f"  CAT micro-avg (avg of all 49 cells)        = {cat_micro*100:.2f}% -> {cat_micro*100:.1f}%")
print(f"  CAT macro-avg (avg of 10 dataset averages) = {cat_macro*100:.2f}% -> {cat_macro*100:.1f}%")
print(f"  Claimed in thesis: 87.0% (matches macro-avg = avg of per-dataset accuracies)")
assert abs(cat_macro * 100 - 87.0) < 0.5, f"CAT 87.0% (macro) mismatch! Got {cat_macro*100:.2f}"
cat_avg = cat_macro

print("\n--- Table 7: ACCsubcat (last column) per dataset (CDT_MSW) ---")
table7_total = {
    "Banana":        0.7467,
    "Titanic":       0.9667,
    "Thyroid":       0.9067,
    "Diabetis":      0.8850,
    "Breast_cancer": 0.9600,
    "Heart":         0.9733,
    "German":        0.9057,
    "KDDCup99":      0.8667,
    "MITface":       0.6632,
    "USPS":          0.7952,
}
sub_avg = np.mean(list(table7_total.values()))
print(f"  SUB average = {sub_avg*100:.1f}%   (claimed in thesis: 86.7%)")
assert abs(sub_avg * 100 - 86.7) < 0.5, f"SUB 86.7% mismatch! Got {sub_avg*100:.2f}"

print("\n--- Table 7: ACCsubcat per drift type per dataset ---")
table7_per_type = {
    "Sudden":      [0.9333, 1.0000, 0.8667, 0.9333, 1.0000, 0.9333, 1.0000, 1.0000, 0.4444, 0.9091],
    "Blip":        [0.8000, 1.0000, 0.7333, 1.0000, 1.0000, 1.0000, 0.9333, 1.0000, 0.7143, 0.6667],
    "Recurrent":   [0.5000, 0.8333, 1.0000, 0.6250, 1.0000, 1.0000, 0.9286, 1.0000, 0.5000, 1.0000],
    "Incremental": [0.5000, 1.0000, 0.9333, 0.8667, 0.8667, 0.9333, 0.6667, 0.3333, 0.8000, 0.5000],
    "Gradual":     [1.0000, 1.0000, 1.0000, 1.0000, 0.9333, 1.0000, 1.0000, 1.0000, 0.8571, 0.9000],
}
expected_per_type = {"Sudden": 90.2, "Blip": 88.5, "Recurrent": 83.9, "Incremental": 74.0, "Gradual": 96.9}
for t, vals in table7_per_type.items():
    avg = np.mean(vals) * 100
    expected = expected_per_type[t]
    print(f"  {t:12s}: {avg:.1f}%   (claimed in thesis: {expected}%)   {'OK' if abs(avg-expected) < 0.5 else 'MISMATCH!'}")
    assert abs(avg - expected) < 0.5, f"{t} mismatch!"

print("\n--- Table 5: MDR per drift type x window size (CDT_MSW) ---")
table5 = {
    "Sudden":      [0.133, 0.033, 0.000, 0.100, 0.067],
    "Blip":        [0.333, 0.033, 0.133, 0.300, 0.100],
    "Recurrent":   [0.434, 0.334, 0.467, 0.634, 0.266],
    "Incremental": [0.133, 0.100, 0.033, 0.100, 0.000],
    "Gradual":     [0.100, 0.033, 0.167, 0.233, 0.100],
}
print("  Per-type MDR averages and Recall = 1 - MDR:")
all_mdr = []
for t, vals in table5.items():
    mdr_avg = np.mean(vals)
    recall = 1 - mdr_avg
    all_mdr.extend(vals)
    print(f"    {t:12s}: MDR avg = {mdr_avg*100:5.1f}%  ->  Recall = {recall*100:5.1f}%")

mdr_micro = np.mean(all_mdr)
mdr_macro = np.mean([np.mean(vs) for vs in table5.values()])
recall_micro = 1 - mdr_micro
recall_macro = 1 - mdr_macro
print(f"  MDR micro-avg (over 25 cells)         = {mdr_micro*100:.2f}% -> Recall = {recall_micro*100:.2f}%")
print(f"  MDR macro-avg (over 5 type averages)  = {mdr_macro*100:.2f}% -> Recall = {recall_macro*100:.2f}%")
print(f"  -> NEW value to use in thesis: Recall ~= {recall_macro*100:.1f}% (macro, rounds to 82.5%)")
overall_recall = recall_macro

print("\n--- Table 4: EDR (Error Detection Rate / FDR-like) per drift type ---")
print("  (Paper definition: EDR = false detections / total detections, lower is better)")
print("  This is NOT comparable to our 'EDR' (Event Detection Rate = Recall)")
table4 = {
    "Sudden":      [0.175, 0.123, 0.115, 0.188, 0.100],
    "Blip":        [0.415, 0.123, 0.200, 0.308, 0.125],
    "Recurrent":   [0.217, 0.199, 0.311, 0.353, 0.133],
    "Incremental": [0.224, 0.148, 0.075, 0.125, 0.042],
    "Gradual":     [0.247, 0.118, 0.050, 0.241, 0.083],
}
all_edr = [v for vals in table4.values() for v in vals]
edr_avg = np.mean(all_edr)
print(f"  Paper EDR avg = {edr_avg*100:.1f}%   (was claimed as '17.8%' in thesis - mathematically right but WRONG metric to compare)")
assert abs(edr_avg * 100 - 17.8) < 0.5

print("\n" + "=" * 80)
print("SUMMARY OF VERIFIED PAPER NUMBERS")
print("=" * 80)
print(f"  CAT (drift category accuracy)  = {cat_avg*100:.1f}%  [Table 6 avg]")
print(f"  SUB (drift subcategory acc)    = {sub_avg*100:.1f}%  [Table 7 avg]")
print(f"  Per-type SUB:")
for t, vals in table7_per_type.items():
    print(f"    {t:12s} = {np.mean(vals)*100:.1f}%  [Table 7 column avg]")
print(f"  Recall = 1 - MDR avg           = {overall_recall*100:.1f}%  [Table 5, NEW correct comparison metric]")
print(f"  Paper 'EDR' (FDR-like)         = {edr_avg*100:.1f}%  [Table 4, OLD wrong-metric comparison]")
print("=" * 80)
