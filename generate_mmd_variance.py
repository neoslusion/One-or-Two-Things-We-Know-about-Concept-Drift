import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_theme(style="whitegrid")

# Generate data
np.random.seed(42)
data_P = np.random.normal(0, 1, 1000)
data_Q = np.random.normal(0, 3, 1000)

fig, ax = plt.subplots(figsize=(8, 5))

# Plot distributions
sns.kdeplot(data_P, fill=True, color='blue', alpha=0.5, label='Distribution P\n(Mean=0, Var=1)')
sns.kdeplot(data_Q, fill=True, color='red', alpha=0.3, label='Distribution Q\n(Mean=0, Var=9)')

# Add mean lines (overlapping at 0)
plt.axvline(x=0, color='black', linestyle='--', linewidth=2, label='Shared Mean ($\mu=0$)')

# Annotations
plt.text(4, 0.2, 'Same Mean,\nDifferent Variance', fontsize=12, 
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5'))

# MMD Annotation
plt.text(4, 0.1, r'$MMD^2 \gg 0$' + '\n(Drift Detected!)', fontsize=14, color='darkred', weight='bold',
         bbox=dict(facecolor='lightyellow', alpha=0.9, edgecolor='red', boxstyle='round,pad=0.5'))

# Styling
plt.title("MMD Detects Variance Shifts (Unlike Mean-based Tests)", fontsize=14, pad=15)
plt.xlabel("Feature Value ($X$)", fontsize=12)
plt.ylabel("Density", fontsize=12)
plt.legend(loc='upper left', fontsize=10)

plt.tight_layout()
plt.savefig('report/latex/image/mmd_variance_change.png', dpi=300, bbox_inches='tight')
print("Saved image to report/latex/image/mmd_variance_change.png")
