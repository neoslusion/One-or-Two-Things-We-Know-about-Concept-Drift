import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns

sns.set_theme(style="whitegrid")

# Simulate null distribution of MMD (which is an infinite sum of Chi-squares, often approximated by Gamma)
np.random.seed(42)
# Generate a skewed distribution (Gamma)
shape, scale = 2.0, 1.5
data = np.random.gamma(shape, scale, 1000)

plt.figure(figsize=(8, 5))
# Plot histogram of simulated null MMD values
sns.histplot(data, bins=30, stat='density', alpha=0.6, color='skyblue', label='Simulated Null MMD$^2$')

# Fit a Gamma distribution
x = np.linspace(0, max(data)+2, 100)
y = stats.gamma.pdf(x, a=shape, scale=scale)
plt.plot(x, y, 'r-', lw=3, label='Gamma Approximation')

# Fill the rejection region (p-value < 0.05)
threshold = stats.gamma.ppf(0.95, a=shape, scale=scale)
x_fill = np.linspace(threshold, max(x), 50)
y_fill = stats.gamma.pdf(x_fill, a=shape, scale=scale)
plt.fill_between(x_fill, y_fill, color='red', alpha=0.3, label=r'Rejection Region ($\alpha=0.05$)')
plt.axvline(threshold, color='red', linestyle='--', lw=2)
plt.text(threshold + 0.2, 0.1, r'Threshold $c_\alpha$', color='red', fontsize=12)

plt.title('Moment-Matched Gamma Approximation for MMD', fontsize=14)
plt.xlabel('MMD$^2$', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend()
plt.tight_layout()
plt.savefig('gamma_approximation.png', dpi=300)
print("Image saved as gamma_approximation.png")
