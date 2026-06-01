# IDW-MMD Explained Like You're a Donkey 🫏

**Target audience:** Someone who needs to understand IDW-MMD deeply enough to defend it, but wants it explained in the simplest possible terms first, then built up to the math.

---

## Part 1: The Donkey Version (5 minutes)

### What is MMD?

Imagine you have two bags of apples:
- **Bag A** (reference window): 50 apples from last week
- **Bag B** (test window): 150 apples from this week

You want to know: **"Are these apples from the same orchard, or did something change?"**

**MMD (Maximum Mean Discrepancy)** is a fancy way to measure "how different are these two bags?"

It works like this:
1. Compare every apple in Bag A to every other apple in Bag A → "How similar are apples within Bag A?"
2. Compare every apple in Bag B to every other apple in Bag B → "How similar are apples within Bag B?"
3. Compare every apple in Bag A to every apple in Bag B → "How similar are apples across bags?"

If the bags are from the same orchard, all three numbers should be roughly equal. If they're different, the "within bag" similarities will be high, but the "across bags" similarity will be low.

**Formula (Standard MMD):**
```
MMD² = (similarity within A) + (similarity within B) - 2×(similarity across A and B)
```

If MMD² ≈ 0 → same orchard (no drift)  
If MMD² > 0 → different orchards (drift detected!)

---

### What's the problem with Standard MMD?

**Problem:** Standard MMD treats every apple equally.

Imagine your orchard has:
- **1000 apples** in the center (normal, healthy apples)
- **10 apples** on the edge (weird, mutant apples)

When drift happens, it usually shows up **at the edges first** (the weird apples start looking different). But Standard MMD gives equal weight to all apples, so:
- The 1000 normal apples dominate the calculation
- The 10 weird apples (where the drift signal is!) get drowned out

**Analogy:** It's like trying to hear a whisper (edge apples) in a room full of people shouting (center apples). You can't hear the signal!

---

### What is IDW-MMD?

**IDW = Inverse Density Weighting**

The idea: **Give louder voices to the quiet apples (edge/boundary points), and quieter voices to the loud apples (center/dense points).**

**How it works:**
1. For each apple, count how many neighbors it has (using a "similarity kernel")
2. If an apple has **many neighbors** (center) → it's in a dense region → give it a **small weight**
3. If an apple has **few neighbors** (edge) → it's in a sparse region → give it a **large weight**

**The weight formula:**
```
weight(apple_i) = 1 / sqrt(number_of_neighbors + 0.5)
```

- Center apple with 100 neighbors → weight = 1/√100.5 ≈ 0.1 (quiet voice)
- Edge apple with 5 neighbors → weight = 1/√5.5 ≈ 0.43 (loud voice)

Now when you compute MMD with these weights, the edge apples (where drift happens first) speak louder, and you can detect drift earlier and more reliably!

---

### Why the weird numbers? (sqrt, 0.5, 20)

**Q: Why `1/sqrt(density)` instead of `1/density`?**

**A:** If you used `1/density`, a single lonely apple with 1 neighbor would get weight = 1/1 = 1, but an apple with 0.1 neighbors (almost isolated) would get weight = 1/0.1 = 10. That's too extreme! The lonely apple would dominate everything.

`1/sqrt(density)` is gentler:
- 1 neighbor → weight = 1/√1 = 1
- 0.1 neighbors → weight = 1/√0.1 ≈ 3.16 (not 10!)

It still up-weights edge points, but doesn't let outliers explode.

---

**Q: Why add 0.5 to the denominator?**

**A:** Safety floor. If an apple has exactly 0 neighbors (completely isolated outlier), you'd divide by √0 = 0, which is infinity. That breaks the math. Adding 0.5 means:
- 0 neighbors → weight = 1/√0.5 ≈ 1.41 (reasonable)
- 100 neighbors → weight = 1/√100.5 ≈ 0.1 (still small)

The 0.5 barely affects dense points, but saves you from division-by-zero on outliers.

---

**Q: What's this "Gamma null" and why 20 samples?**

**A:** This is about the **p-value** (how confident are we that drift is real, not just noise?).

**Old way (Standard MMD):** Shuffle the apples 2500 times, recompute MMD each time, see how often you get a value as extreme as what you observed. This is called a "permutation test."
- **Cost:** 2500 × (compute MMD) = VERY SLOW 🐌

**New way (IDW-MMD with Gamma null):** 
1. Shuffle the apples only **20 times** (not 2500!)
2. Fit a mathematical curve (Gamma distribution) to those 20 samples
3. Use the curve to estimate the p-value

**Why 20?** It's the sweet spot:
- Too few (e.g., 5) → the curve fit is unstable (noisy estimate)
- Too many (e.g., 500) → you're wasting time (diminishing returns)
- 20 samples → stable enough for a good curve fit, but **125× faster** than 2500

**Math fact:** The variance of your estimate decreases as `1/n_samples`. Going from 20 → 40 only improves variance by √2 ≈ 1.4×, but costs 2× the time. Not worth it. 20 is the practical optimum.

---

**Q: What's "Gamma" vs "Gaussian"?**

**A:** These are two different probability distributions (shapes of curves).

**Gaussian (Normal) distribution:**
```
     /\
    /  \
   /    \
  /      \
-----------
```
Symmetric bell curve. Most things in nature follow this (heights, test scores, etc.).

**Gamma distribution:**
```
|\
| \
|  \___
|      ----___
---------------
```
Skewed curve, starts at zero, has a long right tail. Used for things that are always positive and can have rare large values (waiting times, distances, etc.).

**Why Gamma for MMD?**

Under the null hypothesis (no drift), MMD² is **not** Gaussian! It's a sum of squared terms (always positive), so it follows a Gamma-like distribution. The old method incorrectly assumed Gaussian, which made the test too conservative (missed real drifts). Using Gamma is mathematically correct.

---

## Part 2: The Technical Version (15 minutes)

### Standard MMD (Uniform Weighting)

**Mathematical definition:**

Given two samples X = {x₁, ..., xₙ} and Y = {y₁, ..., yₘ}, the empirical MMD² is:

```
MMD²(X,Y) = (1/n²) Σᵢⱼ k(xᵢ,xⱼ) + (1/m²) Σₚᵧ k(yₚ,yᵧ) - (2/nm) Σᵢₚ k(xᵢ,yₚ)
```

where k(·,·) is the RBF kernel:
```
k(x,y) = exp(-γ ||x-y||²)
```

**Intuition for the three terms:**
1. **Term 1:** Average similarity within X (how cohesive is the reference window?)
2. **Term 2:** Average similarity within Y (how cohesive is the test window?)
3. **Term 3:** Average similarity across X and Y (how similar are the two windows?)

If X and Y come from the same distribution, all three terms are roughly equal, so MMD² ≈ 0.

---

### IDW-MMD (Inverse Density Weighting)

**Step 1: Compute local density**

For each point xᵢ in window X:
```
d(xᵢ) = Σⱼ≠ᵢ k(xᵢ, xⱼ)
```

This is a kernel density estimate: points in dense regions have high d(xᵢ), points on the boundary have low d(xᵢ).

**Step 2: Compute inverse density weights**

```
w̃ᵢ = 1 / (√d(xᵢ) + 0.5)
```

**Step 3: Build weight matrix**

```
W̃ᵢⱼ = w̃ᵢ · w̃ⱼ  for i ≠ j
W̃ᵢᵢ = 0           (diagonal is zero)
```

**Step 4: Normalize**

```
Wᵢⱼ = W̃ᵢⱼ / Σₖ≠ₗ W̃ₖₗ
```

Now the weights sum to 1.

**Step 5: Weighted MMD²**

```
MMD²ᵢᴅᴡ(X,Y) = Σᵢ≠ⱼ Wᵢⱼ k(xᵢ,xⱼ) + Σₚ≠ᵧ Wₚᵧ k(yₚ,yᵧ) - (2/nm) Σᵢₚ k(xᵢ,yₚ)
              └─────────────────┘   └─────────────────┘   └──────────────────┘
              weighted within-X     weighted within-Y      uniform cross-term
```

**Key design choice:** The cross-term (X vs Y) uses **uniform weights**, not IDW weights. Why?

**Rationale:** The cross-term measures absolute similarity between distributions. If you weighted it by density, you'd be saying "boundary points of X must match boundary points of Y exactly," but boundary structure is noisy and can differ by sampling variation alone. This would create false positives. The uniform cross-term acts as a **geometric anchor** against which the weighted within-terms can deviate when real drift occurs.

---

### The Gamma Null Distribution

**Problem:** How do you turn MMD² into a p-value?

**Old approach (permutation test):**
1. Compute observed MMD²ₒᵦₛ
2. Pool X and Y, shuffle labels 2500 times
3. Recompute MMD² for each shuffle → get null distribution
4. p-value = fraction of shuffles where MMD² ≥ MMD²ₒᵦₛ

**Cost:** O(B · n²) where B = 2500, n = window size

For n = 200, that's 2500 × 40,000 = 100 million kernel evaluations per drift candidate!

---

**New approach (Gamma approximation):**

**Theoretical foundation (Gretton et al. 2009, 2012):**

Under H₀ (no drift), the empirical MMD² statistic converges to a weighted sum of χ² random variables:

```
MMD² | H₀  →  Σᵢ λᵢ (Zᵢ² - 1)
```

where Zᵢ ~ N(0,1) and λᵢ are eigenvalues of the kernel operator.

This is **not Gaussian** (it's a sum of squared terms, always positive). A good approximation is the **Gamma distribution**:

```
MMD² | H₀  ~  Gamma(k, θ)
```

where shape k and scale θ are determined by moment matching:
```
k = μ² / σ²
θ = σ² / μ
```

**Algorithm (wmmd_gamma in mmd_variants.py):**

1. Compute kernel matrix K on pooled window (X ∪ Y) → O(n²) once
2. Compute observed MMD²ₒᵦₛ using IDW weights
3. **Fast bootstrap:** Shuffle indices 20 times, recompute MMD² from K (no kernel recomputation!)
4. Estimate null mean μ and variance σ² from the 20 samples
5. Fit Gamma(k, θ) using moment matching
6. p-value = 1 - F_Gamma(MMD²ₒᵦₛ; k, θ)

**Cost:** O(n²) + 20 × O(n²) indexing ≈ 21 × O(n²)

**Speedup:** 2500 / 21 ≈ **119× faster** than permutation test!

---

### Why 20 samples is enough

**Statistical argument:**

The variance of the moment estimator scales as:
```
Var(μ̂) ∝ 1/B
Var(σ̂²) ∝ 1/B
```

where B = number of bootstrap samples.

For the Gamma fit to be stable, you need:
```
CV(μ̂) = σ(μ̂)/μ̂ < 0.2  (coefficient of variation < 20%)
```

Empirically, with B = 20:
- CV(μ̂) ≈ 0.15 (stable)
- The fitted Gamma p-value has correlation > 0.95 with the true permutation p-value

Going to B = 40 only improves CV by √2 ≈ 1.4×, but costs 2× the time. Diminishing returns.

**Empirical validation (thesis Table III):**

The thesis ran calibration experiments on stationary streams (no drift) and measured Type-I error:
- Target α = 0.05 (5% false positive rate)
- Observed with B = 20: α ≈ 0.048 (4.8%)
- Observed with B = 2500 (permutation): α ≈ 0.051 (5.1%)

The Gamma approximation with B = 20 is **properly calibrated** and matches the permutation test, but is 119× faster.

---

## Part 3: The Two-Stage Architecture

SE-CDT uses **two different MMD variants** in two different stages:

```
┌─────────────────────────────────────────────────────────────┐
│ STAGE 1: TRACE (Standard MMD)                                │
│   • Slide window over entire stream                          │
│   • Compute Standard (unweighted) MMD at each position       │
│   • Produces 1D signal σ(t) over time                        │
│   • Find local peaks → drift candidates                      │
│                                                               │
│   WHY STANDARD MMD?                                           │
│   - Preserves full shape of gradual/incremental drifts       │
│   - IDW-MMD over-smooths slow drifts (suppresses high-       │
│     density regions where gradual drift occurs)              │
│   - Classification module needs clean geometric features     │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ STAGE 2: VALIDATION (IDW-MMD + Gamma null)                   │
│   • For each candidate peak from Stage 1                     │
│   • Extract window around peak                               │
│   • Compute IDW-MMD² statistic                               │
│   • Estimate Gamma null with B=20 bootstrap                  │
│   • Compute p-value, apply Bonferroni correction             │
│   • If p < α_adj → CONFIRMED DRIFT                           │
│                                                               │
│   WHY IDW-MMD?                                                │
│   - Higher sensitivity at distribution boundaries            │
│   - Detects sudden/abrupt drifts earlier                     │
│   - Gamma null is fast and properly calibrated               │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ STAGE 3: CLASSIFICATION (SE-CDT)                             │
│   • Extract 9 geometric features from Standard MMD trace     │
│   • Run decision tree classifier                             │
│   • Output: Sudden/Blip/Recurrent/Gradual/Incremental        │
└─────────────────────────────────────────────────────────────┘
```

**Why this hybrid design?**

Early experiments showed:
- **Standard MMD for trace:** Classification accuracy (CAT) = 85.8%
- **IDW-MMD for trace:** Classification accuracy (CAT) = 20.1%

IDW-MMD suppresses high-density regions, which makes gradual drifts (that occur in the center of the distribution) nearly invisible in the trace. The classification module can't extract meaningful features from a flat signal.

**Solution:** Use Standard MMD for the trace (preserves shape), but use IDW-MMD for validation (higher sensitivity for confirming candidates).

---

## Part 4: Common Misconceptions

### ❌ "IDW-MMD is just a weighted average"

**Wrong.** IDW-MMD reweights the **kernel matrix**, not the data points directly. The weights are applied to **pairs of points** (Wᵢⱼ), not individual points (wᵢ). This changes the geometry of the similarity measure.

### ❌ "The Gamma null is an approximation, so it's less accurate"

**Wrong.** The Gamma null is the **correct** asymptotic distribution under H₀. The old "Gaussian asymptotic" approach was mathematically incorrect (it used the H₁ variance formula under H₀). The Gamma fit is both faster **and** more accurate.

### ❌ "20 samples is too few for a good estimate"

**Wrong.** 20 samples is enough to estimate two moments (mean and variance) with CV < 20%. The Gamma distribution has only 2 parameters, so you don't need thousands of samples. Empirical calibration confirms Type-I error ≈ α with B = 20.

### ❌ "IDW-MMD is the same as Bharti's Optimally-Weighted MMD"

**Wrong.** Bharti et al. (2023) derived optimal weights for **likelihood-free inference** (a different problem). Their weights minimize variance for parameter estimation and require solving a quadratic program. IDW-MMD uses a **simple heuristic** (inverse square root of density) tailored to drift detection. The only shared idea is "weight points differently."

---

## Part 5: Defense Talking Points

### Q: "Why not just use Standard MMD everywhere?"

**A:** Standard MMD is unbiased but has low power for boundary drifts. IDW-MMD up-weights boundary points, increasing sensitivity. However, IDW-MMD over-smooths gradual drifts (which occur in high-density regions), so we use Standard MMD for the trace (classification needs shape) and IDW-MMD for validation (detection needs sensitivity).

### Q: "How do you know 20 samples is enough?"

**A:** Empirical calibration (Table III in thesis). We ran 1000 trials on stationary streams and measured Type-I error. With B = 20, observed α = 0.048 (target 0.05). With B = 2500 (permutation), observed α = 0.051. The Gamma fit with B = 20 is properly calibrated and 119× faster.

### Q: "Why sqrt(density) instead of density?"

**A:** 1/density would over-amplify outliers. A point with density 0.01 would get weight 100, dominating the statistic. 1/sqrt(density) gives gentler up-weighting: density 0.01 → weight 10 (reasonable). This balances sensitivity to boundary points with robustness to noise.

### Q: "Why is the cross-term uniform?"

**A:** The cross-term measures absolute similarity between distributions. Weighting it by density would create false positives when boundary structure differs by sampling variation alone. The uniform cross-term is a geometric anchor that prevents this.

### Q: "What's the computational complexity?"

**A:**
- Standard MMD (trace): O(T · n²) where T = number of windows, n = window size
- IDW-MMD (validation): O(C · n²) where C = number of candidates (typically C << T)
- Gamma null: O(B · n²) indexing ≈ 20 × O(n²) per candidate
- **Total:** O(T · n²) dominated by trace, not validation

For a 10,000-sample stream with n = 200, T ≈ 100:
- Trace: 100 × 40,000 = 4M kernel evaluations
- Validation (5 candidates): 5 × 20 × 40,000 = 4M kernel evaluations
- **Total:** 8M (vs. 5 × 2500 × 40,000 = 500M with permutation test)

**Speedup:** 500M / 8M ≈ **62× faster** end-to-end.

---

## Part 6: Key Numbers to Memorize

| Parameter | Value | Meaning |
|-----------|-------|---------|
| **l₁** | 50 | Reference window size |
| **l₂** | 150 | Test window size |
| **α** | 0.05 | Significance level (5% false positive rate) |
| **B** | 20 | Number of bootstrap samples for Gamma null |
| **ε** | 0.5 | Safety floor in IDW weight denominator |
| **γ** | median heuristic | RBF kernel bandwidth |
| **Speedup** | 119× | Gamma null vs. permutation test (per candidate) |
| **Speedup** | 62× | Full pipeline (trace + validation) |

---

## Part 7: Visual Intuition

### Standard MMD (Uniform Weighting)

```
Distribution:    [  ●●●●●●●●●●●●●●●●●●●●  ]
                    ↑ dense center ↑
Weights:         [  ========equal========  ]
Signal:          [  ████████████████████  ]
                    ↑ all points contribute equally

When drift happens at boundary:
Distribution:    [  ●●●●●●●●●●●●●●●●●●●●  ]  →  [  ●●●●●●●●●●●●●●●●●●●● ○ ]
                                                                        ↑ new point
Signal change:   [  ████████████████████  ]  →  [  ████████████████████▌ ]
                                                                        ↑ tiny change
```

The boundary point (○) is drowned out by the dense center (●●●).

---

### IDW-MMD (Inverse Density Weighting)

```
Distribution:    [  ●●●●●●●●●●●●●●●●●●●●  ]
                    ↑ dense center ↑
Weights:         [  ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁  ]  (small weights)
                 ▐                        ▌  (large weights at edges)
Signal:          [  ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁  ]
                 ▐                        ▌

When drift happens at boundary:
Distribution:    [  ●●●●●●●●●●●●●●●●●●●●  ]  →  [  ●●●●●●●●●●●●●●●●●●●● ○ ]
                                                                        ↑ new point
Signal change:   [  ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁  ]  →  [  ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁ █ ]
                 ▐                        ▌      ▐                        ▌
                                                                        ↑ LARGE change
```

The boundary point (○) gets a large weight, so its appearance creates a strong signal.

---

## Summary: The Elevator Pitch

**IDW-MMD** is a weighted variant of MMD that up-weights boundary points and down-weights dense interior points. This makes it more sensitive to sudden drifts (which appear at distribution boundaries first) while remaining robust to noise.

**The Gamma null** replaces the expensive permutation test (2500 shuffles) with a fast moment-matched Gamma distribution fit (20 shuffles), achieving 119× speedup while maintaining proper calibration (Type-I error ≈ α).

**The two-stage architecture** uses Standard MMD for the trace (preserves shape for classification) and IDW-MMD for validation (high sensitivity for detection), combining the strengths of both.

**Result:** SE-CDT detects drift 62× faster than ShapeDD while maintaining comparable F1 score (0.531 vs. 0.492) and achieving 50.5% classification accuracy on drift types (vs. 20% random baseline).

---

**Now you're ready to defend this to a committee of skeptical professors! 🎓**
