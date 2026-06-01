# Defense Preparation Guide

**Quick Start for Thesis Defense**

---

## 📚 Essential Documents (Read These)

### 1. **[THESIS_GUIDE.md](./THESIS_GUIDE.md)** - Complete Technical Reference
**Purpose:** Comprehensive guide covering everything from problem definition to Q&A  
**When to read:** Week 1-4 for deep understanding  
**Time:** 3 hours

### 2. **[DEFENSE_PREP_VN.md](./DEFENSE_PREP_VN.md)** - Vietnamese Defense Guide
**Purpose:** Vietnamese language preparation materials  
**When to read:** For Vietnamese-language defense preparation  
**Time:** 1-2 hours

---

## 🎯 Quick Reference (Use These for Review)

### **IDW-MMD Quick Reference Card**

**30-Second Elevator Pitch:**
> "IDW-MMD up-weights boundary points (where drift appears first) and down-weights dense center points. Combined with a fast Gamma-null p-value (20 samples instead of 2500), it achieves 119× speedup per validation while maintaining proper calibration (Type-I error ≈ 0.05)."

**Key Numbers to Memorize:**
| Parameter | Value | Meaning |
|-----------|-------|---------|
| l₁ | 50 | Reference window size |
| l₂ | 150 | Test window size |
| α | 0.05 | Significance level |
| B | 20 | Bootstrap samples for Gamma null |
| ε | 0.5 | Safety floor in IDW weight |
| Speedup | 119× | Gamma vs permutation (per validation) |
| Speedup | 7× | End-to-end pipeline |
| F1 | 0.531 | Detection performance |
| CAT | 50.5% | Classification accuracy |

**The Three Core Formulas:**
```
1. Local Density:  d(xᵢ) = Σⱼ≠ᵢ k(xᵢ,xⱼ)
2. IDW Weight:     wᵢ = 1/(√d(xᵢ) + 0.5)
3. IDW-MMD²:       Σᵢⱼ Wᵢⱼ k(xᵢ,xⱼ) + Σₚᵧ Wₚᵧ k(yₚ,yᵧ) - (2/nm) Σᵢₚ k(xᵢ,yₚ)
                   └─────weighted─────┘   └─────weighted─────┘   └────uniform────┘
```

---

## 🤔 Common Defense Questions

### Q1: "What's the main contribution?"
**A:** SE-CDT - unified detector-classifier with two innovations:
1. Detection: IDW-MMD + Gamma null (7× faster, properly calibrated)
2. Classification: Unsupervised drift-type identification (50.5% accuracy)

### Q2: "Why not use Standard MMD everywhere?"
**A:** Standard MMD drowns out boundary points where drift appears first. IDW-MMD up-weights boundaries (+30% sensitivity). But IDW-MMD over-smooths gradual drifts, so we use Standard MMD for trace (classification needs shape) and IDW-MMD for validation (detection needs sensitivity).

### Q3: "How do you know 20 samples is enough?"
**A:** Three pieces of evidence:
1. Statistical theory: Var(estimator) ∝ 1/B, so B=20 gives CV ≈ 15%
2. Empirical calibration: Type-I error = 0.048 (target: 0.05)
3. Diminishing returns: B=20→40 only improves by √2 but costs 2×

### Q4: "Why is Gradual/Incremental accuracy so low (30.8% / 4.4%)?"
**A:** Fundamental limitation of unsupervised discrimination. CDT-MSW (96.9%/74.0%) uses supervised features (accuracy curves, labels). SE-CDT only sees MMD trace, which is ambiguous. However:
- Both Gradual and Incremental use the same adaptation strategy (continuous update)
- The critical distinction is TCD vs PCD (for adaptation), not Gradual vs Incremental
- SE-CDT achieves strong TCD detection: Sudden 82.4%, Recurrent 71.5%

### Q5: "Why is the cross-term uniform?"
**A:** 
- **Intuitive:** Boundary points are random and differ between samples even from the same distribution. Weighting them creates false positives from sampling noise.
- **Mathematical:** Uniform weights give an unbiased estimator: E[(1/nm)Σk(xᵢ,yₚ)] = E[k(X,Y)]. Weighted cross-term introduces bias: E[ΣWᵢₚ·k] ≠ E[k] because Wᵢₚ depends on the sample, creating Cov(W,k) ≠ 0.

### Q6: "Is this the same as Bharti's Optimally-Weighted MMD?"
**A:** No. Bharti et al. (2023) derived optimal weights for likelihood-free inference (different problem). IDW-MMD uses a simple heuristic (inverse sqrt of density) tailored to drift detection. Only shared idea: "weight points differently."

---

## 📖 Deep Dive Topics (If Asked)

### **Gradual vs Incremental: Formal Definitions**

**Gradual Drift (Gama et al. 2014):**
- Probabilistic mixture of two concepts
- Formula: `P_t = (1-α(t))·P_old + α(t)·P_new` where α increases linearly
- Each sample drawn from old OR new concept
- Implementation: `if random() < α: sample from new else: sample from old`

**Incremental Drift (Webb et al. 2016):**
- Continuous parameter evolution
- Formula: `θ_t = θ_0 + v·t` (single evolving distribution)
- Each sample from current shifted distribution
- Implementation: `X[t] = randn() + (magnitude × progress)`

**Key Difference:**
- Gradual: Discrete concept space (2 concepts), probabilistic switching
- Incremental: Continuous concept space (infinite states), deterministic drift

**Why SE-CDT struggles:**
- Both produce "wide peaks" in MMD trace
- Temporal features (LTS, MS, SDS) are noisy
- Cannot reliably distinguish from distribution distance alone
- CDT-MSW succeeds because it uses supervised accuracy curves

---

### **IDW-MMD Algorithm (Step-by-Step)**

**Input:** X (n=50 samples), Y (m=150 samples), γ (bandwidth), ε=0.5

**Step 1:** Compute kernel matrix K_XX
```
K_XX[i,j] = exp(-γ × ||xᵢ-xⱼ||²)
```

**Step 2:** Compute local density (off-diagonal sum)
```
d(xᵢ) = Σⱼ≠ᵢ K_XX[i,j]
```
High d → dense center, Low d → boundary

**Step 3:** Compute inverse density weights
```
w̃ᵢ = 1/(√d(xᵢ) + 0.5)
```
Why sqrt? Gentler up-weighting (1/d would over-amplify outliers)  
Why 0.5? Safety floor to prevent division by zero

**Step 4:** Build pairwise weight matrix
```
W̃ᵢⱼ = w̃ᵢ × w̃ⱼ  for i≠j
W̃ᵢᵢ = 0         (diagonal is zero)
```

**Step 5:** Normalize
```
Wᵢⱼ = W̃ᵢⱼ / Σₖₗ W̃ₖₗ
```

**Step 6:** Compute weighted within-X term
```
Term1 = Σᵢⱼ Wᵢⱼ × K_XX[i,j]
```

**Step 7:** Repeat for Y → Term2

**Step 8:** Compute uniform cross-term
```
Term3 = (2/nm) Σᵢₚ k(xᵢ,yₚ)
```

**Step 9:** Combine
```
MMD²_IDW = Term1 + Term2 - Term3
MMD_IDW = √max(0, MMD²_IDW)
```

---

### **Why Gamma, Not Gaussian?**

**Under H₀ (no drift), MMD² is:**
- A sum of squared terms (always positive)
- Right-skewed distribution
- Approximately Gamma-distributed

**Gaussian distribution:**
```
     /\
    /  \      ← Symmetric, can be negative
   /    \
  /      \
```

**Gamma distribution:**
```
|\
| \
|  \___       ← Right-skewed, always positive
|      ----___
```

**Why Gamma is correct:**
- MMD² = sum of weighted χ² variables (Gretton et al. 2012)
- Gamma is the correct asymptotic distribution under H₀
- Old "Gaussian asymptotic" used H₁ variance under H₀ (incorrect!)

**Empirical validation:**
- Gamma with B=20: Type-I error = 0.048 (target: 0.05) ✓
- Properly calibrated at α = 0.05

---

## ✅ Defense Checklist

### **1 Week Before**
- [ ] Read THESIS_GUIDE.md Sections 1, 4, 5, 12
- [ ] Memorize key numbers (table above)
- [ ] Practice answering Q1-Q6
- [ ] Review all figures and tables in thesis

### **3 Days Before**
- [ ] Review this quick reference card
- [ ] Practice 30-second elevator pitch
- [ ] Review common misconceptions
- [ ] Test presentation slides

### **1 Day Before**
- [ ] Read this document (10 minutes)
- [ ] Review key formulas
- [ ] Practice top 5 questions
- [ ] Get good sleep!

### **Day Of**
- [ ] Quick review of key numbers (5 minutes)
- [ ] Breathe and stay confident
- [ ] Remember: You know this better than anyone!

---

## 🎓 Final Tips

**Be honest about limitations:**
- Gradual/Incremental discrimination is hard (unsupervised vs supervised)
- Acknowledge it, explain why, point to future work

**Emphasize practical impact:**
- 7× speedup over state-of-the-art
- Real-time Kafka deployment
- Proper statistical calibration

**Show intellectual honesty:**
- Proper attribution (Bharti, ShapeDD, CDT-MSW)
- Statistical rigor (H₀ calibration, Friedman test)
- Honest evaluation (report all metrics, not just best)

**Stay calm:**
- If you don't know, say "That's a good question, I'd need to investigate further"
- Don't make up answers
- Redirect to what you do know

---

**You've got this! 💪🎓**

---

## 📂 File Structure

```
/home/goldship/sandboxes/One-or-Two-Things-We-Know-about-Concept-Drift/
├── THESIS_GUIDE.md              ← Complete technical reference (1933 lines)
├── DEFENSE_PREP_VN.md           ← Vietnamese defense guide
├── README_DEFENSE.md            ← This file (quick reference)
└── report/latex/                ← LaTeX thesis source
```

**For detailed explanations, see THESIS_GUIDE.md sections:**
- Section 1: The problem
- Section 4: IDW-MMD (detection module)
- Section 5: SE-CDT (classification module)
- Section 12: Anticipated Q&A
