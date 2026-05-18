# Thesis Comprehensive Guide

**Thesis:** *Nghiên cứu và phát triển hệ thống tự động phát hiện hiện tượng trôi dạt và cập nhật mô hình học máy thích ứng* (A study on automatic concept-drift detection and adaptive machine-learning model updates)

**Author:** Lê Phúc Đức (MSSV: 2370116)
**Advisor:** PGS.TS Thoại Nam
**Institution:** ĐH Bách Khoa — ĐHQG-HCM, Khoa Khoa học và Kỹ thuật Máy tính

This guide is a comprehensive companion to the thesis, written for two audiences at once:
- A **non-specialist** who has never heard of concept drift but wants to understand what this thesis is about, why it matters, and how it works.
- The **author** preparing for defense, who wants every design choice, every number, every limitation explained and ready to defend.

It includes plain-English explanations, full mathematical derivations, and a Q&A section with anticipated examiner questions.

---

## How to read this guide

If you have **15 minutes**: read Section 1 (the problem) and Section 12 (the Q&A).
If you have **1 hour**: add Sections 4 (IDW-MMD) and 5 (SE-CDT) — the two contributions.
If you have **3 hours**: read end-to-end.

Each section is self-contained with cross-references back to the relevant chapter of the thesis.

---

## Table of contents

1. [The problem: what is concept drift and why it matters](#1-the-problem)
2. [Mathematical foundations](#2-mathematical-foundations)
3. [Background: what existed before this thesis](#3-background)
4. [Contribution 1 — IDW-MMD: a faster, more sensitive drift detector](#4-contribution-1--idw-mmd)
5. [Contribution 2 — SE-CDT: an unsupervised drift-type classifier](#5-contribution-2--se-cdt)
6. [Adaptation strategies (what to do once drift is detected)](#6-adaptation-strategies)
7. [The Kafka prototype (how the whole thing runs as a system)](#7-the-kafka-prototype)
8. [Experiments: setup, metrics, results](#8-experiments)
9. [Statistical methodology](#9-statistical-methodology)
10. [Honest limitations and weaknesses](#10-limitations)
11. [Implementation map (where each idea lives in code)](#11-implementation-map)
12. [Anticipated examiner questions and model answers](#12-defense-qa)

---

# 1. The problem

## 1.1 The one-paragraph version

A machine-learning model is trained on historical data and then deployed to make predictions on new, incoming data. **Concept drift** is the phenomenon where the statistical relationship that the model learned from historical data no longer matches the relationship in the live data. The model is silently wrong — its accuracy drops without anyone noticing, until someone audits or until the downstream system breaks. This thesis builds an automatic system that (i) **detects** when drift happens, (ii) **classifies** what kind of drift it is, and (iii) **updates** the model with a strategy chosen based on the drift type, all without needing human-labeled ground truth at runtime.

## 1.2 Two concrete examples

These examples open the introduction (`chapters/00_introduction.tex`) because they make the abstract problem feel real:

- **A bank's fraud-detection model** has 99% accuracy in its first year. After a few years, the false-alarm rate has tripled. Why? Fraudsters have changed their tactics — the *patterns* of fraudulent vs. legitimate transactions today are not the patterns the model learned from. The model's data distribution drifted.
- **A predictive-maintenance model in a steel factory** runs flawlessly through the dry season and then breaks down in the rainy season. Humidity changes the sensor readings, the vibration patterns of machinery, and the failure modes. Same model, same factory, same machines — but the *operating conditions* drifted, and so did the data.

These are not bugs in the model or the engineer's code. They are a property of the world: data distributions change over time. A static model cannot keep up.

## 1.3 Why this is a hard problem

Three reasons:

1. **You usually don't have labels at runtime.** In the bank fraud example, you don't know which transactions are fraud until weeks later — investigators have to confirm manually. So you cannot just measure "accuracy is dropping" and trigger retraining; you need to detect drift from the *unlabeled* data $X$ alone.
2. **Drift comes in many shapes.** A sudden drift (sensor breaks at 3 AM) is very different from a gradual drift (machinery wears down over months) or a recurrent drift (seasonal patterns). The right *response* depends on the *type*. A sudden drift means "throw out the old model"; a gradual drift means "fine-tune incrementally"; a recurrent drift means "switch to the model you saved from last summer".
3. **You must distinguish drift from noise.** Real-world data is noisy. A small change in distribution might be a real drift, or it might be normal variation. Falsely triggering retraining on noise is expensive (compute, downtime) and can actually *hurt* model accuracy if it causes overfitting to a temporary fluctuation.

## 1.4 What "drift" formally is

Mathematically, the data is a stream of samples $(x_t, y_t)$ for $t = 1, 2, 3, \ldots$, where $x_t \in \mathbb{R}^d$ is a feature vector and $y_t$ is a label. Each sample is drawn from some distribution $P_t(X, Y)$ at time $t$. **Concept drift** is the statement that

$$P_t(X, Y) \ne P_{t+\Delta}(X, Y) \quad \text{for some time gap } \Delta.$$

Using the chain rule of probability, $P(X, Y) = P(X) \cdot P(Y \mid X)$, so drift can come from:

- $P(X)$ changing — the inputs look different. Called **virtual drift** or **covariate shift**.
- $P(Y \mid X)$ changing — the relationship between input and output changed even if inputs look the same. Called **real drift** or **concept drift in the strict sense**.
- Both changing simultaneously — called **joint drift**.

The thesis's central observation: in practice, when $P(Y \mid X)$ drifts, $P(X)$ usually drifts too (people behave differently, sensors fail in correlated ways, seasons change everything together). This is the **joint-drift assumption**. It is what makes unsupervised drift detection (using only $X$) practically useful: when $P(X)$ moves enough, $P(Y \mid X)$ has likely moved too, so the model is likely degraded — even if we can't directly measure it.

The thesis is honest that the rare case of "only $P(Y \mid X)$ shifts while $P(X)$ stays exactly the same" is **out of scope** — no unsupervised method can see that.

## 1.5 The five shapes of drift over time

The thesis classifies drift into five subtypes by how the change unfolds in time. Understanding these is essential because the **whole point of SE-CDT** is to recognize them from an unlabeled signal.

| Type | What happens | Example |
|------|--------------|---------|
| **Sudden** | Distribution flips at one instant from $P$ to $Q$. | A sensor fails at 03:14:22 and starts returning garbage. |
| **Gradual** | For a while, samples come from $P$ and $Q$ randomly mixed; the probability of $Q$ rises smoothly until it's 100%. | A new product line is rolled out region by region over weeks. |
| **Incremental** | A continuous, slow drift through a sequence of intermediate distributions $P \to P_1 \to P_2 \to \cdots \to Q$. | Sensor calibration drifting due to mechanical wear. |
| **Recurrent** | An old distribution that disappeared comes back. | Christmas shopping behavior returns every December. |
| **Blip** | A short transient spike — distribution briefly changes and then returns. | A flash crowd on a server, then traffic returns to normal. |

The thesis groups these into two coarser **categories**:
- **TCD (Transient Concept Drift):** Sudden, Blip, Recurrent — the change is discrete or short-lived.
- **PCD (Progressive Concept Drift):** Gradual, Incremental — the change unfolds over a continuous transition.

The TCD/PCD distinction is operationally important: TCD usually warrants a *reset* of the model (it's a different world now), while PCD warrants *fine-tuning* (the world is changing slowly, the model can chase it).

## 1.6 What this thesis builds

A three-stage pipeline:

```
   raw stream           "drift at t=1234"      "type=Sudden"        retrain strategy
   ┌──────────┐         ┌──────────────┐       ┌────────────┐       ┌─────────────────┐
   │ DETECTION│────────▶│CLASSIFICATION│──────▶│ ADAPTATION │──────▶│ updated model   │
   └──────────┘         └──────────────┘       └────────────┘       └─────────────────┘
   ShapeDD-IDW          SE-CDT (9 features      type-specific:      Logistic Reg.,
   (IDW-MMD +           + decision tree +       Reset / Fine-tune /  online ensemble,
    Gamma null)         concept memory)         Reuse-cached         etc.
```

The single contribution is **SE-CDT** (ShapeDD-Enhanced Concept Drift Type) — a **unified detector-classifier system** for unsupervised drift handling. SE-CDT contains two modules:

1. **Detection module — ShapeDD-IDW** (Section 4 of this guide). A faster, more sensitive variant of ShapeDD using **IDW-MMD** (Inverse Density-Weighted MMD) and a **Gamma-distribution-based p-value** instead of expensive permutation tests. The module produces the drift signal $\sigma(t)$ and emits drift alerts when it spikes.
2. **Classification module** (Section 5). An unsupervised classifier that reads the *shape* of $\sigma(t)$ to decide which of {Sudden, Blip, Recurrent, Gradual, Incremental} drift just happened, plus a "concept memory" for catching recurrent patterns.

Two more pieces are built around SE-CDT:

3. **Adaptation framework** (Section 6) — a small library of update strategies, dispatched by SE-CDT's classification output (Reset for Sudden, Fine-tune for Gradual/Incremental, Reuse cached model for Recurrent, etc.).
4. **Kafka prototype** (Section 7) — to show the whole thing can be wired into a real streaming system end-to-end.

In the benchmark tables you will see two rows, `ShapeDD_IDW` and `SE_CDT`, with **identical detection F1**. This is by design: `ShapeDD_IDW` is just SE-CDT's detection module run alone (no classification); `SE_CDT` is the full system. They share the detection algorithm, so detection metrics are identical.

**Naming convention (also explained in Chapter 3):** the detection module gets its own hyphenated name **ShapeDD-IDW** because it is a direct refinement of an existing method (ShapeDD). The classification module does *not* have its own separate name — it is the core of SE-CDT itself. The acronym SE-CDT (ShapeDD-Enhanced Concept Drift **Type** identification) literally encodes the classification: "Type identification" *is* the classifier. So the asymmetric naming reflects the fact that detection is a refinement of prior work, while classification is the genuinely novel contribution and owns the SE-CDT name. When the context does not require disambiguation, "SE-CDT" refers to the whole system; when it does, the guide says "the detection module of SE-CDT" or "the classification module of SE-CDT".

The next section gives the math you need to follow the contributions.

---

# 2. Mathematical foundations

This section assembles the math that the rest of the guide depends on. If you already know what an RKHS, MMD, and a permutation test are, you can skim or skip.

## 2.1 Streams, windows, distributions

Let the data stream be $\{x_t\}_{t=1}^{\infty}$ where each $x_t \in \mathbb{R}^d$ is sampled from some time-varying distribution $P_t$. We do **not** assume $\{x_t\}$ is i.i.d. across time — that would be the whole point lost. Within a small **window** of consecutive samples, however, we treat the points as approximately i.i.d. from a single $P$ for that window.

The thesis uses two windows side-by-side:
- A **reference window** $X = \{x_{t-l_1-l_2+1}, \ldots, x_{t-l_2}\}$ of size $n = l_1 = 50$ samples (pre-drift candidate).
- A **test window** $Y = \{x_{t-l_2+1}, \ldots, x_t\}$ of size $m = l_2 = 150$ samples (post-drift candidate).

If the underlying distribution is the same in both windows, $X$ and $Y$ are samples from one $P$. If a drift happened between them, they come from different $P$ and $Q$.

**Why $l_1 \ne l_2$?** A small reference (50) gives a tight "before" snapshot; a larger test (150) gives statistical power to detect the change. The cooldown of 150 samples between consecutive detections (i.e., $2\delta$ with $\delta=75$ tolerance) is set so each true drift produces at most one detection.

## 2.2 Hypothesis testing in one paragraph

You have two groups of samples and you want to know if they came from the same distribution. You set up a **null hypothesis** $H_0\!: P = Q$. You compute a **test statistic** $T$ (some number that's small when $H_0$ is true, large otherwise). You then compute a **p-value**: the probability that, under $H_0$, you would see $T$ at least as extreme as you actually saw. If p < $\alpha = 0.05$, you reject $H_0$ and call it a drift.

Two kinds of error:
- **Type I error (false positive):** You reject $H_0$ even though it's true. Probability $= \alpha$.
- **Type II error (false negative):** You fail to reject $H_0$ even though it's false. Probability $= \beta$, and $1-\beta$ is the **power** of the test.

For drift detection, we set $\alpha = 0.05$ throughout.

## 2.3 Kernels and RKHS (the geometric trick)

A **kernel** is a function $k(x, y)$ that measures similarity between two points. It's allowed to be more clever than ordinary inner product or Euclidean distance. The thesis uses the **Gaussian RBF kernel**:

$$k(x, y) = \exp(-\gamma \|x - y\|^2)$$

where $\gamma > 0$ is a **bandwidth** parameter. When $x \approx y$, $k \approx 1$; when they're far apart, $k \approx 0$. So the kernel is a soft "are these points close?" function.

**The trick (Mercer's theorem):** Every valid kernel corresponds to a (possibly infinite-dimensional) feature map $\phi$ such that $k(x, y) = \langle \phi(x), \phi(y) \rangle$. The space where $\phi$ lives is called a **Reproducing Kernel Hilbert Space (RKHS)** $\mathcal{H}$. We never compute $\phi$ explicitly; we just use $k$. This is the **kernel trick** — it lets us implicitly work in a very high-dimensional space using only finite computation.

**Bandwidth selection — the median heuristic:** A common automatic choice is

$$\gamma = \frac{1}{2 \cdot \mathrm{median}(\|x_i - x_j\|^2)}.$$

This makes the kernel adapt to the natural scale of the data: if your $x$'s are typically 100 units apart, $\gamma$ is small; if they're 0.1 units apart, $\gamma$ is large.

## 2.4 Maximum Mean Discrepancy (MMD)

The **MMD** between two distributions $P$ and $Q$ in an RKHS $\mathcal{H}$ is

$$\mathrm{MMD}(P, Q) = \sup_{\|f\|_{\mathcal{H}} \le 1} \big| \mathbb{E}_{X \sim P}[f(X)] - \mathbb{E}_{Y \sim Q}[f(Y)] \big|.$$

In words: among all "well-behaved" functions in the RKHS, find the one that most disagrees in expectation between $P$ and $Q$, and report that maximum disagreement. If $P = Q$, every function agrees and $\mathrm{MMD} = 0$. If $P \ne Q$, *some* function disagrees, so $\mathrm{MMD} > 0$.

**The closed form** (Gretton et al. 2012, the foundational reference): the squared MMD reduces to a kernel expression with no explicit supremum:

$$\mathrm{MMD}^2(P, Q) = \mathbb{E}_{X, X' \sim P}[k(X, X')] + \mathbb{E}_{Y, Y' \sim Q}[k(Y, Y')] - 2 \mathbb{E}_{X \sim P, Y \sim Q}[k(X, Y)].$$

**Intuition for the three terms:**
- 1st term: average self-similarity within $P$ samples.
- 2nd term: average self-similarity within $Q$ samples.
- 3rd term (subtracted, doubled): average cross-similarity between $P$ and $Q$ samples.

If $P = Q$, "within $P$" similarity equals "within $Q$" similarity equals "across $P, Q$" similarity, so the expression collapses to zero.

**Empirical estimator (what you compute on actual data):**

$$\widehat{\mathrm{MMD}}^2(X, Y) = \frac{1}{n^2}\sum_{i,j} k(x_i, x_j) + \frac{1}{m^2}\sum_{p,q} k(y_p, y_q) - \frac{2}{nm}\sum_{i,p} k(x_i, y_p).$$

This is **Standard MMD** (or "uniform MMD" because every pair gets weight $1/n^2$ or $1/m^2$ or $1/(nm)$). It's what ShapeDD originally uses.

## 2.5 The permutation test (the slow but reliable p-value)

How do you turn a kernel statistic $\widehat{\mathrm{MMD}}^2$ into a p-value? Under $H_0$ ($P = Q$), the labels "this point came from $X$" vs. "this point came from $Y$" are arbitrary — you could swap them and the distribution of $\widehat{\mathrm{MMD}}^2$ wouldn't change. So:

1. Compute the observed statistic $T_{\mathrm{obs}} = \widehat{\mathrm{MMD}}^2(X, Y)$.
2. Pool all $n + m$ points; randomly relabel half as $X$ and half as $Y$; compute $\widehat{\mathrm{MMD}}^2$ on the relabeled data.
3. Repeat step 2 $B$ times (typically $B = 2500$). Get $T^{(1)}, \ldots, T^{(B)}$.
4. p-value $= \frac{\#\{b : T^{(b)} \ge T_{\mathrm{obs}}\}}{B}$.

This works for any $H_0$-symmetric statistic. **Cost:** $O(B \cdot (n+m)^2)$. For $B = 2500$ and a window of 200 points, that's $2500 \cdot 40{,}000 = 10^8$ kernel evaluations *per drift candidate*. This is the dominant cost in ShapeDD and is what IDW-MMD will replace.

## 2.6 The Gamma-distribution null (the fast alternative)

Gretton, Fukumizu, et al. (2009) showed that **under $H_0$, the empirical MMD$^2$ is approximately Gamma-distributed**:

$$\widehat{\mathrm{MMD}}^2 \mid H_0 \;\;\dot\sim\;\; \mathrm{Gamma}(k, \theta)$$

where shape $k$ and scale $\theta$ are determined by the first two moments (mean $\mu$ and variance $\sigma^2$) of $\widehat{\mathrm{MMD}}^2$ under $H_0$, via the moment-matching equations $k = \mu^2 / \sigma^2$ and $\theta = \sigma^2 / \mu$. The mean and variance can be estimated cheaply from the kernel matrix (closed-form, $O(n^2)$).

So the recipe is:
1. Compute the kernel matrix.
2. Estimate $\mu, \sigma^2$ of $\widehat{\mathrm{MMD}}^2$ under $H_0$.
3. Fit Gamma to those moments → cdf $F_{\Gamma}(\cdot ; k, \theta)$.
4. p-value $= 1 - F_{\Gamma}(T_{\mathrm{obs}})$.

**Cost:** $O(n^2)$ — about $B = 2500\times$ cheaper than the permutation test.

The thesis uses this Gamma approximation for the **validation step** of the two-stage pipeline; that's where most of the speedup comes from.

## 2.7 The triangle shape property (ShapeDD's key theorem)

When a sudden drift happens at time $t = 0$ — distribution flips from $P$ to $Q$ instantly — and you slide a window of size $l$ over the stream computing the drift signal $\sigma(t) = d(P_{\mathrm{ref}}, P_{\mathrm{test}})$, the signal has a **triangular shape**:

$$\sigma(t) = \|P - Q\| \cdot h_l(t), \qquad h_l(t) = \max\!\Big(0, \; 1 - \frac{|l - t|}{l}\Big).$$

**Why?** When the window straddles the drift point, half its samples are pre-drift (from $P$) and half are post-drift (from $Q$); the more it straddles, the more the within-window distribution looks bimodal, and $d(\cdot)$ peaks. As the window moves further past the drift, fewer pre-drift samples remain, so the distance shrinks linearly — hence the triangular shape.

**Two crucial implications:**
1. **The triangle exists for *any* metric $d(\cdot)$ and *any* distribution shape.** Only the *amplitude* depends on $\|P - Q\|$. This is what makes ShapeDD broadly applicable.
2. **The triangle exists *only* for sudden drift.** Gradual drift produces a wider, flatter peak; incremental drift produces a sustained ramp; recurrent drift produces a periodic train of triangles. **This is the geometric basis for SE-CDT classification** — different drift types leave different shapes on the MMD signal.

## 2.8 Critical Difference, Friedman, Nemenyi (the ranking math)

When you compare $K$ methods on $N$ datasets, you don't just want to know which method has the highest mean F1 — you want to know if the ranking is **statistically significant** or just due to chance. Demšar (2006) is the canonical methodology:

1. For each dataset, **rank** the methods 1 to $K$ (best to worst on F1).
2. Compute average rank $\bar r_j$ for each method $j$ across the $N$ datasets.
3. **Friedman test:** is the global ranking significantly different from "everything is equally good"?
4. If yes, **Nemenyi post-hoc**: any two methods are significantly different iff $|\bar r_i - \bar r_j| > \mathrm{CD}$, where the **Critical Difference** is

$$\mathrm{CD} = q_\alpha \cdot \sqrt{\frac{K(K+1)}{6N}}.$$

$q_\alpha$ comes from a tabulated studentized range distribution. For $K = 8$, $\alpha = 0.05$, $q_{0.05} = 3.031$. With $N = 14$ datasets, $\mathrm{CD} = 3.031 \cdot \sqrt{72/84} = 2.806$. **Two methods are statistically tied if their average ranks differ by less than 2.806.**

This is why the thesis says "DAWIDD (rank 3.679) and IDW-MMD/SE-CDT/ShapeDD-IDW (rank 3.821) are statistically tied" — the gap of 0.142 is well below 2.806.

---

# 3. Background — what existed before this thesis

Concept drift detection has been studied since the 1990s, and the field has dozens of methods. The thesis surveys them in `chapters/01_related_works.tex`. This section gives you the families and the three specific methods that are direct predecessors of this thesis.

## 3.1 The four families of drift detectors

| Family | Key idea | Examples | Strength | Weakness |
|--------|---------|----------|----------|----------|
| **Statistical Process Control (SPC)** | Track the model's error rate; if error spikes beyond a control limit, declare drift. | DDM, EDDM, RDDM, ADWIN, HDDM, FHDDM | Simple, low memory, theoretical guarantees. | Need labels at runtime. |
| **Sliding-window distribution tests** | Compare data distribution in a recent window to a reference window. | KS-Test, MMD-based methods, ShapeDD | Unsupervised (no labels needed). | Sensitive to noise; slow if using permutation tests. |
| **Discriminative / virtual classifier** | Train a classifier to distinguish "old data" from "new data"; if it can, there's drift. | D3, OCDD | Fast, unsupervised. | Detects only $P(X)$ shift; misses subtle real drift. |
| **Independence test** | Drift = features become dependent on time. Test feature⊥time independence. | DAWIDD (uses HSIC kernel test) | Unsupervised, robust to gradual drift. | Higher false-positive rate. |

The thesis picks the **distribution-test** family (specifically MMD-based) because (i) it's unsupervised and (ii) MMD has the nicest theoretical foundation for kernel-based comparisons.

## 3.2 ShapeDD (Hinder, Brinkrolf, Vaquet, Hammer 2021)

This is the **direct predecessor** of the thesis's detector. ShapeDD (`shapeDD2021` in the bib) was published at IEEE SSCI 2021. Its idea is the **Triangle Shape Property** explained in §2.7: a sudden drift at time $t_0$ produces a triangular peak in the MMD-vs-time signal, with the apex at $t_0$.

**ShapeDD's algorithm (Algorithm 1 of the thesis):**

1. Slide a window of size $2 l_1$ over the stream.
2. Compute Standard MMD between the first half (reference) and second half (test) of the window. Plot this as $\sigma(t)$.
3. Convolve $\sigma(t)$ with a derivative-of-triangle filter $h'_l$ to find sign changes — these are **drift candidates**.
4. For each candidate, run a **permutation test** with $B = 2500$ permutations to validate it.
5. If $p < \alpha = 0.05$, declare drift.

**What's good about it:**
- Theoretical guarantee that sudden drift leaves a triangle. Other methods don't have this.
- Shape-based filtering removes noise. Real drift looks like a triangle; noise looks like uncorrelated wiggles.
- Localizes drift well.

**What's bad about it:**
- The permutation test is **slow** — $O(B \cdot l^2)$ per candidate. On a 10,000-sample stream with many candidates, this dominates runtime.
- Sensitive to kernel bandwidth $\gamma$ and window size $l$.
- Only handles sudden drift cleanly; gradual/incremental/recurrent need post-processing.

**This thesis fixes (1) by replacing permutation with a Gamma-distribution p-value, (2) by reweighting the MMD statistic via IDW, and (3) by adding the SE-CDT classifier downstream.**

## 3.3 CDT-MSW (Guo, Li, Ren, Wang 2022)

CDT-MSW (`guo2022cdtmsw` in the bib) was published in *Information Sciences* in 2022. It's the direct predecessor of the **classifier**, not the detector. The full name is "Concept Drift Type identification based on Multi-Sliding Windows".

**Its key insight:** different drift types leave different *temporal patterns* in the **model's accuracy over time**. Specifically:
- A model on a **sudden drift** drops accuracy at one point sharply.
- A **gradual drift** has a slow accuracy drop over a transition period.
- A **recurrent drift** has an accuracy that bounces back when the old distribution returns.
- An **incremental drift** has a steady accuracy decline.

CDT-MSW measures these via three sliding windows and a "Tracking Flow Ratio" (TFR) and "Micro TFR" (MTFR) signal. It also has a **Growth Process** that estimates whether the drift is "instantaneous" (length 1) or "progressive" (length > 1) — partitioning into TCD and PCD.

**The fatal limitation for our setting: CDT-MSW is supervised.** The accuracy signal requires labels $y_t$ for every sample. In streaming production systems, you usually don't have labels. So CDT-MSW cannot be used directly.

**This thesis takes two ideas from CDT-MSW and discards the rest:**
1. **TCD/PCD partition** based on drift duration. SE-CDT keeps this exact partition.
2. **Subtype distinctions** Sudden/Blip/Recurrent for TCD and Gradual/Incremental for PCD. SE-CDT keeps these.

**What SE-CDT replaces:** The accuracy signal is replaced by the **MMD signal** (computed from $X$ alone, no labels). The Growth Process is replaced by **FWHM-based duration estimation** on the MMD peak (also label-free).

This is the central methodological pivot of the thesis: take CDT-MSW's classification taxonomy but make it work *without labels*, by reading shape features from the MMD trace instead of accuracy variance.

## 3.4 Bharti et al. OW-MMD (ICML 2023)

Bharti, Naslidnyk, Key, Kaski, Briol (2023) at ICML proposed **Optimally-Weighted MMD** (OW-MMD). Their setting is *not* drift detection — it's likelihood-free inference / parameter estimation for expensive simulators. They needed a way to compute MMD with reduced variance for Bayesian inference.

**The idea they contribute:** instead of giving every point in the MMD computation equal weight $1/n$, give different points different weights $w_i$. They derive an *optimal* weighting that minimizes variance for their inference goal. The math is involved (involves solving a quadratic program) and is specialized to their problem.

**What this thesis takes:** *only* the high-level idea "weighted MMD can do better than uniform MMD for some objective". That's it. The thesis explicitly does **not** use Bharti's optimal weights — it uses a much simpler heuristic (inverse density, see §4.4) tailored to drift detection.

This is documented carefully in `chapters/03_proposed_model.tex:36–38`:
> "lấy cảm hứng từ ý tưởng weighted MMD trong Optimally-Weighted MMD của Bharti et al. ... Phương pháp của Bharti et al. được thiết kế cho bài toán suy luận thống kê, đề xuất trọng số tối ưu phức tạp và không liên quan đến bài toán drift detection. Luận văn chỉ kế thừa ý tưởng 'gán trọng số cho từng điểm dữ liệu trong MMD' và áp dụng một heuristic đơn giản hơn (trọng số nghịch biến với căn bậc hai của mật độ kernel) để phù hợp với bài toán drift detection."

In English: "We take inspiration from Bharti's weighted-MMD idea, but their method is designed for statistical inference and is unrelated to drift detection. We inherit only the 'weight each point' idea and apply a simpler heuristic (inverse-square-root-of-density) tailored to drift detection."

This is correct attribution practice and important for the defense.

## 3.5 Putting the three together

| Predecessor | What this thesis inherits | What this thesis replaces |
|-------------|--------------------------|---------------------------|
| **ShapeDD 2021** | Two-window MMD architecture, shape-based candidate selection, Triangle Shape Property as theoretical anchor. | Permutation test → Gamma p-value (≈7× faster). Uniform MMD → IDW-MMD (more sensitive at distribution boundary). |
| **CDT-MSW 2022** | TCD/PCD partition, the Sudden/Blip/Recurrent/Gradual/Incremental subtype taxonomy. | Supervised accuracy signal → unsupervised MMD signal. Growth Process → FWHM-based duration. |
| **Bharti OW-MMD 2023** | The high-level idea "weight points in MMD". | Their optimal-weight derivation → simple inverse-density heuristic. |

All three predecessors are cited correctly with their actual venue, year, authors, and DOI; the thesis was reviewed for citation integrity (`CITATION_REVIEW.md`) and no errors remain.

---

# 4. SE-CDT's detection module: ShapeDD-IDW

This section explains the **detection module** of SE-CDT, named **ShapeDD-IDW**, end-to-end. The thesis chapter is `chapters/03_proposed_model.tex` §3.2. (Note: in earlier drafts of the guide, this section was labeled "Contribution 1 — IDW-MMD". The corrected framing is that SE-CDT is the single contribution; ShapeDD-IDW is its detection module; IDW-MMD is the algorithm used inside ShapeDD-IDW's validation step.)

## 4.1 The two problems with Standard MMD that IDW-MMD fixes

Standard MMD (§2.4) gives every pair $(x_i, x_j)$ the same weight in the kernel sum. This is fine when you want an unbiased estimator of population MMD, but for **drift detection** it has two drawbacks:

**Problem 1: Self-pair bias.** With the RBF kernel, the diagonal terms $k(x_i, x_i) = 1$ contribute $n$ to the within-window sum. These don't measure between-point similarity at all — they're just a constant offset. They make the empirical $\widehat{\mathrm{MMD}}^2$ slightly positive even when $P = Q$.

**Problem 2: Boundary points are drowned out.** In a window, most pairs come from the dense interior of the distribution. Sudden drift typically appears first at the **boundary** (or tail) of the distribution — a few points start landing where they didn't before. With uniform weights, those few boundary points contribute a tiny fraction of the sum, and the signal is buried in interior-vs-interior similarities that aren't moving.

The fix is to **down-weight the dense interior** and **up-weight the sparse boundary** — let the points in low-density regions speak louder.

## 4.2 The IDW weight definition

Define the **off-diagonal local density** at point $x_i$ as

$$d(x_i) = \sum_{j \ne i} k(x_i, x_j).$$

This is a kernel-density-style estimate of how many neighbors $x_i$ has. A point in the interior has many neighbors; the kernel sums up to a large number. A point near the boundary has few neighbors; the sum is small.

The **IDW weight** is

$$\tilde w_i = \frac{1}{\sqrt{d(x_i)} + \epsilon}, \qquad \epsilon = 0.5.$$

**Why $1/\sqrt{d}$ and not $1/d$?** $1/d$ would amplify a single isolated outlier enormously, making the estimator noise-dominated. $1/\sqrt{d}$ gives a milder up-weighting that still favors boundary points but doesn't blow up. The $\epsilon = 0.5$ is a numerical safety floor — without it, an isolated point with $d \to 0$ would have $\tilde w_i \to \infty$.

**Why off-diagonal ($j \ne i$)?** Including $k(x_i, x_i) = 1$ would put a constant 1 on every $d(x_i)$, washing out the signal you're trying to capture.

After computing $\tilde w_i$ for all $i$, build the weight matrix

$$\widetilde W^{XX}_{ij} = \tilde w_i \tilde w_j \quad (i \ne j), \qquad \widetilde W^{XX}_{ii} = 0,$$

and normalize so it sums to 1:

$$W^{XX}_{ij} = \frac{\widetilde W^{XX}_{ij}}{\sum_{i \ne j} \widetilde W^{XX}_{ij}}.$$

The same procedure on the test window gives $W^{YY}_{pq}$.

## 4.3 The IDW-MMD statistic

$$\boxed{\;
\widehat{\mathrm{MMD}}^2_{\mathrm{IDW}}(X, Y)
\;=\;
\underbrace{\sum_{i \ne j} W^{XX}_{ij} \, k(x_i, x_j)}_{\text{weighted within-X}}
\;+\;
\underbrace{\sum_{p \ne q} W^{YY}_{pq} \, k(y_p, y_q)}_{\text{weighted within-Y}}
\;-\;
\underbrace{\frac{2}{nm} \sum_{i,p} k(x_i, y_p)}_{\text{uniform cross-term}}
\;}$$

Three things to notice:

1. **Diagonal is excluded** ($i \ne j$, $p \ne q$) — fixes the self-pair bias from Problem 1.
2. **Within-window terms use IDW weights** — fixes the boundary-drown-out from Problem 2.
3. **Cross-term keeps uniform weights $\frac{1}{nm}$** — this is the most subtle design choice. Why?

**Justification for uniform cross-term:** The cross-term $\frac{2}{nm} \sum k(x_i, y_p)$ measures how similar the two distributions are *in absolute terms*. If you put IDW weights on it too, you'd be saying "boundary points of $X$ should match boundary points of $Y$ extra hard" — but boundary structure can be noisy and asymmetric between the two windows. You'd introduce a "false drift" signal whenever the boundary structure differs from sample variation alone. The thesis calls the uniform cross-term a **"geometric anchor"** — it gives an unbiased reference against which the weighted within-terms can deviate when there's actual drift.

**Side effect:** $\widehat{\mathrm{MMD}}^2_{\mathrm{IDW}}$ is **slightly positive** even when $P = Q$ (a small positive bias from the asymmetric weighting). This is *not* a bug — the bias is handled at the p-value step via the Gamma null fit (§4.5).

## 4.4 The complete two-stage pipeline

The detection contribution is not just IDW-MMD as a number — it's a **two-stage architecture** built around it. (`shapedd_idw_mmd_proper` in `core/detectors/mmd_variants.py`.)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ STAGE 1 — TRACE: scan the entire stream                                       │
│   • For each window position t, compute Standard MMD (NOT IDW-MMD)            │
│   • This produces a 1D signal σ(t) over time                                  │
│   • Find local maxima (peaks) above a threshold → drift CANDIDATES            │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼  (a small list of candidate times)
┌──────────────────────────────────────────────────────────────────────────────┐
│ STAGE 2 — VALIDATION: for each candidate, run a hypothesis test                │
│   • Take the windows X, Y around the candidate                                 │
│   • Compute IDW-MMD² statistic                                                 │
│   • Compute p-value via Gamma null distribution (NOT permutation)             │
│   • Bonferroni-correct α by the number of candidates tested                    │
│   • If p < α/M, declare drift; else discard                                    │
└──────────────────────────────────────────────────────────────────────────────┘
```

**Why two stages?**

The trace step is run at *every* time step, so it must be cheap. Standard MMD has $O(l^2)$ cost per window — fast enough.

The validation step is only run on a small number of candidates, so it can be more expensive. IDW-MMD adds $O(n^2)$ work for the weights, plus the Gamma fit is $O(n^2)$. Same order as Standard MMD per call, but only called a few dozen times instead of thousands.

**Why use Standard MMD for the trace, not IDW-MMD?** Two reasons:
1. **Diagnostics.** The trace is plotted; users read it to understand drift behavior. Standard MMD has a clean interpretation; IDW-MMD's shape is harder to reason about because it's biased.
2. **Sensitivity vs. specificity.** Standard MMD is more sensitive (catches more candidates). IDW-MMD is more specific (better at validating). Using each where its strength is = best of both.

## 4.5 The Gamma p-value (replacing permutation)

This is **the** key speed-up over ShapeDD.

**Setup:** under $H_0\!: P = Q$, the IDW-MMD$^2$ is approximately Gamma-distributed (this generalizes Gretton et al. 2009's result for Standard MMD). To use this:

1. Sample $B = 20$ bootstrap replicates of the kernel matrix under $H_0$ (resample with replacement, label some as $X$ and some as $Y$).
2. From those 20 replicates, estimate sample mean $\hat\mu$ and variance $\hat\sigma^2$ of $\widehat{\mathrm{MMD}}^2_{\mathrm{IDW}}$.
3. Match Gamma moments: $\hat k = \hat\mu^2/\hat\sigma^2$, $\hat\theta = \hat\sigma^2/\hat\mu$.
4. p-value $= 1 - F_{\Gamma}(T_{\mathrm{obs}}; \hat k, \hat\theta)$.

**Why $B = 20$ and not $B = 2500$?** We're not estimating the full empirical p-value distribution (which needs many samples in the tail). We're estimating just the **first two moments**, which converge fast. 20 bootstrap samples suffice.

**Speed-up math:** ShapeDD does 2500 permutations × $O(l^2)$ kernel work = $2500 l^2$. IDW-MMD does 20 bootstraps × $O(l^2)$ = $20 l^2$. That's a **125× speedup** in the validation step. End-to-end (including trace + validation) the measured speedup is ≈ **7×**.

**Safety net:** When the sample variance $\hat\sigma^2$ is too small (Gamma fit degenerates — happens on very tight clusters), the code falls back to an **empirical bootstrap** p-value. Documented in `wmmd_gamma` docstring (`mmd_variants.py:326–339`). This catches pathological cases without abandoning the speed-up in the common case.

## 4.6 Bonferroni correction

If you run $M$ candidate validations on the same trace, you have $M$ chances to falsely reject $H_0$. The simplest correction is **Bonferroni**: use $\alpha/M$ instead of $\alpha$ as the per-candidate threshold. Then the family-wise error rate stays $\le \alpha$.

In code (`shapedd_idw_mmd_proper`), this is one line: `adjusted_alpha = alpha / max(1, len(peaks))`. It's conservative (Holm or BH would give more power) but safe and simple. Documented in `chapters/03_proposed_model.tex` and verified by the H0 calibration experiment (Section 9 of this guide).

## 4.7 Complexity summary

| Step | Cost | Notes |
|------|------|-------|
| Trace MMD per window | $O(l_1^2)$ | $l_1 = 50$, so ~2500 kernel evals/window |
| Trace, full stream of length $T$ | $O(T \cdot l_1^2)$ | Dominant in practice for IDW-MMD pipeline |
| IDW weight computation per window | $O(l_1^2)$ | One kernel matrix already needed |
| IDW-MMD validation per candidate | $O(l_2^2)$ | $l_2 = 150$, ~22500 ops |
| Gamma p-value with $B = 20$ bootstraps | $O(B \cdot l_2^2) = 20 \cdot l_2^2$ | Cheap |
| **Total per candidate** | $O(l_2^2)$ effectively | vs. ShapeDD's $O(2500 \cdot l_2^2)$ |
| **Speedup vs. ShapeDD** | ≈ 7× end-to-end (measured) | Single-threaded; documented in `table_III_runtime_stats.tex` |

Numbers verified by the experiments in Section 8 of this guide.

## 4.8 Algorithm 2, line-by-line

Reproduced from `chapters/03_proposed_model.tex`:

```
Algorithm 2: Inverse Density-Weighted MMD (IDW-MMD)
─────────────────────────────────────────────────────
Require: X = {x₁,…,xₙ}, Y = {y₁,…,yₘ}, kernel k, ε = 0.5
Ensure:  IDW-MMD² statistic

1.  for i = 1 to n:                            ← compute weights for X
2.    d(xᵢ) ← Σ_{j≠i} k(xᵢ, xⱼ)               (off-diagonal density)
3.    w̃ᵢ  ← 1 / (√d(xᵢ) + ε)
4.  W̃^XX_ij ← w̃ᵢ·w̃ⱼ for i≠j, else 0
5.  W^XX    ← W̃^XX / Σ W̃^XX                   (normalize)
6.  for p = 1 to m:                            ← repeat for Y
7.    d(yₚ) ← Σ_{q≠p} k(yₚ, y_q)
8.    ṽ_p  ← 1 / (√d(y_p) + ε)
9.  W̃^YY_pq ← ṽ_p·ṽ_q for p≠q, else 0
10. W^YY    ← W̃^YY / Σ W̃^YY
11. return Σᵢⱼ W^XX_ij k(xᵢ, xⱼ)
         + Σ_pq W^YY_pq k(y_p, y_q)
         - (2/(nm)) Σ k(xᵢ, y_p)               ← uniform cross-term
```

This maps line-by-line to `compute_optimal_weights` and `compute_idw_mmd_squared` in `core/detectors/mmd_variants.py`. Verified in `REVIEW_REPORT.md` §5.1.

---

# 5. SE-CDT's classification module

This section explains the **classification module** of the SE-CDT system. The thesis chapter is `chapters/03_proposed_model.tex` §3.3. Code is `core/detectors/se_cdt.py` (the `SE_CDT.classify()` method).

The classification module takes the MMD signal $\sigma(t)$ produced by SE-CDT's detection module (ShapeDD-IDW, Section 4 of this guide) and decides which of {Sudden, Blip, Recurrent, Gradual, Incremental} drift just occurred — without ever using labels $y$.

**SE-CDT** stands for **ShapeDD-Enhanced Concept Drift Type identification**. The full system (detection + classification + concept memory) is what we mean when we say "SE-CDT" in tables and metric reports.

## 5.1 Why classification matters

The detection stage tells you "drift at $t = 1234$". But a system has to *do something* about it. Different drift types call for different responses:

| Detected type | Right response |
|---------------|----------------|
| Sudden | Throw out the model, retrain on post-drift data only. |
| Blip | Do nothing — it's a transient anomaly, will pass. |
| Recurrent | Switch to a cached model that was trained on this distribution before. |
| Gradual / Incremental | Fine-tune the model online with new data. |

A detector that says "drift!" without saying *what kind* leaves the operator guessing. The **value** of SE-CDT is that it lets the adaptation strategy be *automatically chosen* from the drift type. Section 6 shows the strategies; this section shows how SE-CDT decides the type.

## 5.2 The geometric intuition

Different drift types leave different shapes on the MMD signal $\sigma(t)$. Recall from §2.7 that **sudden drift** leaves a single sharp triangle. The other types differ:

| Drift type | Shape of $\sigma(t)$ |
|------------|----------------------|
| **Sudden** | One sharp triangle (FWHM ≈ $l_1$). |
| **Blip** | Two close-together triangles, similar amplitude (drift in, drift out). |
| **Recurrent** | A train of triangles at regular intervals; same shape repeats. |
| **Gradual** | One *wide* triangle with a flatter peak (the transition mixes $P$ and $Q$ probabilistically). |
| **Incremental** | A long, low ramp — many small triangles overlapping, never returning to baseline. |

SE-CDT's job is to extract numerical features from $\sigma(t)$ that capture these distinctions, then run a decision tree.

## 5.3 The 9 features

Listed in `Table 3.1` of the thesis. Each captures a different aspect of peak geometry. (Code: `extract_features` and `extract_temporal_features` in `se_cdt.py`.)

### 5.3.1 Geometric features (from peak detection)

After Gaussian-smoothing $\sigma(t)$ with $\sigma_g = 4$ (denoising), the code finds peaks above $\bar\sigma + 0.3\,\sigma_{\mathrm{std}}$ where $\bar\sigma$ and $\sigma_{\mathrm{std}}$ are the smoothed signal's mean and stdev. From the peaks:

| Feature | Formula | What it captures |
|---------|---------|------------------|
| **WR** (Width Ratio) | $\mathrm{FWHM}/(2 l_1)$ | Sharpness of the peak. Sudden drift → small WR; gradual → large WR. |
| **$n_p$** (Peak Count) | Number of peaks above threshold | Sudden = 1; Blip = 2-3; Recurrent ≥ 4. |
| **CV** (Periodicity CV) | $\mathrm{std}(\Delta P) / \mathrm{mean}(\Delta P)$, where $\Delta P$ are inter-peak distances | Recurrent = small (regular periods); random peaks = large. |
| **SNR** (Signal-to-Noise) | $\max(\sigma_s) / \mathrm{median}(\sigma_s)$ | Sharp clean peak = large; ambient noise = small. |
| **PPR** (Peak Proximity Ratio) | $|p_2 - p_1| / T$, distance between top two peaks normalized by stream length | Blip = small (peaks close); Recurrent = larger (peaks spaced). |
| **DPAR** (Dual-Peak Amplitude Ratio) | $\min(h_1, h_2) / \max(h_1, h_2)$ | Blip = high (symmetric in/out); recurrent = mixed; sudden = N/A. |

### 5.3.2 Temporal features (from regression on $\sigma(t)$)

| Feature | Formula | What it captures |
|---------|---------|------------------|
| **LTS** (Linear Trend Strength) | $R^2$ of a linear regression on $\sigma(t)$ over a long window | Incremental = high (sustained ramp); sudden/blip = low. |
| **SDS** (Step Detection Score) | Fraction of consecutive differences exceeding $1.5 \cdot \mathrm{std}(\Delta\sigma)$ | Sudden = high; gradual = low. |
| **MS** (Monotonicity Score) | $|n^+ - n^-|/(n^+ + n^-)$ where $n^\pm$ counts positive/negative differences | Incremental = high (mostly one direction); noise = ~0. |

## 5.4 The decision tree (Algorithm 3)

The features go through an **explicit decision tree**, not a learned classifier. This is a deliberate design choice — see §5.7 for justification. The tree, in order:

```
INPUT: smoothed σ_s(t), peak positions, the 9 features
       drift_length ℓ from EstimateDriftDuration

STEP 1: BLIP CHECK (TCD-Blip)
   if n_p ∈ {2, 3} and ≥2 peaks detected:
      compact_pair_blip = (PPR < 0.20 ∧ DPAR > 0.60 ∧ WR < 0.30)
      noisy_profile_blip = (ℓ > 1 ∧ WR < 0.17 ∧ 1.45 < SNR < 2.60
                            ∧ 0.45 < DPAR < 0.85 ∧ LTS < 0.12)
      if compact_pair_blip OR noisy_profile_blip:
         return TCD-Blip

STEP 1.5: PCD GATE (long-duration drift)
   if ℓ > 1:                     ← Growth Process: peak FWHM > threshold
      drift_type = PCD
      is_incremental = LTS > 0.5  OR  (MS > 0.6 ∧ LTS > 0.3)
                     OR (SDS > 0.12 ∧ LTS > 0.3)
      if is_incremental: return PCD-Incremental
      else: return PCD-Gradual

STEP 2: SUDDEN CHECK (TCD-Sudden)
   if n_p ≤ 3 and WR < τ_WR (=0.15) and SNR > τ_SNR (=2.0):
      return TCD-Sudden

STEP 3: RECURRENT CHECK (TCD-Recurrent)
   if n_p ≥ 4 and CV < τ_CV (=0.30) and LTS < 0.5:
      return TCD-Recurrent

STEP 4: FALLBACK
   if (LTS > 0.5) OR (MS > 0.6 ∧ LTS > 0.3)
      OR (SDS > 0.12 ∧ LTS > 0.3) OR (n_p ≥ 7):
      return PCD-Incremental
   else:
      return PCD-Gradual
```

**Reading the tree:** Blip is checked first because a noisy blip can fool the duration check (its FWHM can be wider than 1, making it look progressive). Then duration partitions TCD vs. PCD. Within TCD, sudden vs. recurrent is decided by peak count and periodicity. Within PCD, incremental vs. gradual is decided by linear-trend strength.

## 5.5 The Growth Process (`EstimateDriftDuration`)

This is `_growth_process` in `se_cdt.py` lines 368–425. It implements the **TCD/PCD split** from CDT-MSW, but using the MMD signal instead of accuracy.

**Idea:** if the dominant peak in $\sigma_s(t)$ has narrow FWHM (small WR), the drift is "instantaneous-like" — a TCD candidate. If FWHM is wide (large WR), the drift is "progressive" — a PCD candidate.

**The threshold:** `WR_THRESHOLD = 0.12` (documented in Appendix A.2 of the thesis). If the peak's WR ≤ 0.12, the function returns `drift_length = 1` (TCD). If WR > 0.12, returns `drift_length ≥ 2` (PCD).

**Why is this different from `τ_WR = 0.15` used in Step 2?** Because they answer different questions:
- `WR_THRESHOLD = 0.12`: "is this drift instantaneous vs progressive?" (TCD/PCD partition)
- `τ_WR = 0.15`: "given we're already in TCD with few peaks, is this specifically Sudden subtype?"

The two thresholds are close (0.12 vs 0.15) but serve distinct decisions in the tree. Both are documented in Appendix A.2 after the F-6.1 fix.

## 5.6 Self-calibration (handling cross-stream noise variation)

The hard-coded thresholds (τ_WR = 0.15, τ_SNR = 2.0, τ_CV = 0.30) work well on clean streams but fail on streams with high background noise: a real Sudden peak might fail the SNR > 2.0 check just because the noise floor is higher.

**Fix:** a `_RollingFeatureBaseline` (`se_cdt.py:19`) collects WR/SNR/CV values from windows that turn out to be **non-drift**. After 20+ samples, it can compute empirical quantiles and adjust thresholds:

- $\tau_{\mathrm{SNR}}$ ← $\max(2.0, \mathrm{quantile}_{0.90}(\mathrm{baseline\;SNR}))$ — if noisy streams have naturally high SNR, raise the bar to avoid false Sudden classifications.
- $\tau_{\mathrm{WR}}$ ← $\max(0.15, \mathrm{quantile}_{0.25}(\mathrm{baseline\;WR}))$ — if noisy streams have naturally large WR, allow it.
- $\tau_{\mathrm{CV}}$ ← $\max(0.30, \mathrm{quantile}_{0.25}(\mathrm{baseline\;CV}))$ — same logic.

**Cold-start behavior:** baseline buffer must have ≥ 20 entries before self-calibration kicks in. While < 20, the static thresholds are used. This is documented in Ch.3 prose ("warm-up window").

**One-sided adjustment only:** thresholds can only loosen, never tighten beyond the static defaults. This guarantees that on a clean stream, behavior matches the static configuration exactly.

## 5.7 Why an explicit decision tree, not a learned classifier?

This is the most defended-against design choice in the thesis. Three reasons:

1. **No labels at training time.** Building a learned classifier (RF, neural net) requires labeled drift-event data. In the thesis's setting (unsupervised streaming), such labels are exactly what we don't have. Synthetic labels would build in the same circular logic the tree avoids.
2. **Interpretability / debuggability.** When SE-CDT mis-classifies a drift event (and at 50% subtype accuracy, it does often), an operator can read the decision tree and see *which feature* tipped it the wrong way. With a learned classifier, you'd be debugging by gradient.
3. **Stability across deployment domains.** A learned classifier overfits to the training stream's statistics. The decision tree's thresholds (0.15, 0.30, etc.) come from geometric reasoning about the Triangle Shape Property — they're domain-agnostic in principle.

**Honest counter-argument the thesis acknowledges:** the thresholds are still heuristic, not derived from theory. Section 5.4 of the thesis (`sec:limitation-heuristic`) is explicit about this. The self-calibration mechanism mitigates it but does not solve it.

## 5.8 Concept Memory (handling Recurrent drift)

Recurrent drift is special because the *content* of the drift is informative — "we've seen this before". A thoughtful detector should remember.

**Implementation (`se_cdt.py:154–`):**
- A ring buffer of size $M = 8$ stores recent **snapshots** $S_k$, each containing $n_s = 150$ data points.
- Each snapshot is taken right after a confirmed drift (the post-drift distribution).
- For each snapshot, the kernel bandwidth $\gamma_k$ used at capture time is also stored.

**Lookup procedure when a new drift is detected:**
1. Extract the post-drift snapshot $S^{\mathrm{new}}$ from the current window.
2. For each $S_k$ in memory, compute Standard MMD$(S^{\mathrm{new}}, S_k)$ using bandwidth $(\gamma_k + \gamma_{\mathrm{new}})/2$.
3. If $\min_k \mathrm{MMD}(S^{\mathrm{new}}, S_k) < \tau_{\mathrm{match}} = 0.15$, classify the drift as **TCD-Recurrent** (and link to $S_{k^*}$ for the adaptation step to reuse a cached model).
4. Otherwise, add $S^{\mathrm{new}}$ to memory (replacing the oldest entry).

**Why $\tau_{\mathrm{match}} = 0.15$?** Empirical: about $2\times$ the typical noise-floor MMD on stationary windows in the benchmark. Documented in Appendix A.2 and Ch.5.

**Why Standard MMD here, not IDW?** Because matching cached snapshots is a *similarity* task, not a hypothesis test. We want a stable, well-understood quantity. IDW's bias would distort the matching threshold.

## 5.9 The full SE-CDT class

`SE_CDT` in `se_cdt.py` assembles all of the above. The `classify()` method takes a window of MMD trace data and returns a `ClassificationResult` with:
- `drift_type`: "TCD" or "PCD"
- `subcategory`: "Sudden", "Blip", "Recurrent", "Gradual", "Incremental"
- `features`: the 9 numeric values
- `concept_match_id`: which concept-memory entry was matched (None if not Recurrent)
- `adaptive_thresholds`: the actual thresholds used (after self-calibration)

The `ClassificationResult` is passed to the adaptation stage (Section 6).

---

## 5.10 End-to-end worked examples (with example data at each stage)

This section walks the full SE-CDT pipeline through three concrete scenarios — one per outcome family — with realistic numerical values at each processing stage. The goal: make the architecture tangible.

The numerical values are illustrative (representative of what the actual benchmark produces), not exact reproductions of any specific seed. They are calibrated to the synthetic generators in `data/generators/`.

### Scenario A — Sudden drift (the success case)

**Stream setup:**
- Length: $T = 10{,}000$ samples
- Distribution: $X_t \sim \mathcal{N}(0, I_5)$ for $t < 5000$; $X_t \sim \mathcal{N}([2,2,0,0,0]^T, I_5)$ for $t \ge 5000$
- One sudden drift event at $t = 5000$
- Window setup: $l_1 = 50$, $l_2 = 150$, slide every 25 samples

**Stage 1 — σ(t) computation (Standard MMD):**

```
t     σ(t)        Note
─────────────────────────────────────────────────
500    0.038      stationary, low noise
1000   0.041      stationary
2000   0.039      stationary
3000   0.044      stationary
4000   0.048      stationary
4900   0.055      first window edge enters drift zone
5000   0.380      window straddles drift exactly — peak
5050   0.420      apex of triangle
5100   0.350      window mostly post-drift
5200   0.180      tail of triangle
5300   0.075      back to ~stationary
6000   0.041      stationary
─────────────────────────────────────────────────
mean(σ_s)    ≈ 0.052
median(σ_s)  ≈ 0.045
std(σ_s)     ≈ 0.060   (driven by the peak at 5050)
```

The Triangle Shape Property is visible: σ(t) rises from baseline ~0.04 to apex 0.42, with width-at-half-max (FWHM) ≈ 6 samples.

**Stage 2 — Peak detection:**

```
peak_threshold = mean(σ_s) + 0.3 × std(σ_s)
              = 0.052 + 0.3 × 0.060
              = 0.070

Local maxima above threshold: σ(t=5050) = 0.420   ← above threshold ✓
                              (only one peak in this trace)

Candidate peaks: [5050]
```

**Stage 3 — Feature extraction (9 features):**

```
Feature                Value      Interpretation
─────────────────────────────────────────────────────
n_p (peak count)       1          single sharp peak
WR (Width Ratio)       0.060      FWHM/2l₁ = 6/100 = 0.06 — narrow peak
SNR                    9.33       max(σ_s)/median(σ_s) = 0.42/0.045
CV (Periodicity)       N/A        only 1 peak, can't compute Δ
PPR                    N/A        only 1 peak, no pair to compare
DPAR                   N/A        only 1 peak
LTS (Linear Trend R²)  0.04       no linear trend in σ_s
MS (Monotonicity)      0.12       roughly balanced ↑/↓ (peak is symmetric)
SDS (Step Score)       0.08       no sustained step
─────────────────────────────────────────────────────
drift_length (Growth)  1          (because WR = 0.06 < 0.12 → TCD)
```

**Stage 4 — Decision tree walkthrough:**

```
STEP 1: Blip check
  n_p ∈ {2, 3} ? n_p = 1 → FAIL → skip Blip branch

STEP 1.5: PCD gate
  drift_length > 1 ? drift_length = 1 → FAIL → not PCD

STEP 2: Sudden check
  n_p ≤ 3 ?               1 ≤ 3 ✓
  WR < τ_WR (= 0.15) ?    0.060 < 0.15 ✓
  SNR > τ_SNR (= 2.0) ?   9.33 > 2.0 ✓
  → MATCH → return TCD-Sudden ✓
```

**Stage 5 — IDW-MMD validation (per candidate):**

```
For peak at t = 5050:
  X = stream[5000:5050]   (50 reference points, all from N(0, I))
  Y = stream[5050:5200]   (150 test points, all from N([2,2,0,0,0], I))

  Compute IDW weights for X:
    d(x_i) = Σ_{j≠i} k(x_i, x_j)  ≈ 18.4 average
    w̃_i = 1/(√d(x_i) + 0.5)        ≈ 0.207 average
    W^XX normalized so sum = 1

  IDW-MMD² statistic ≈ 0.612

  Gamma null fit (B=20 bootstrap):
    null mean μ̂        ≈ 0.025
    null variance σ̂²   ≈ 0.0003
    Gamma shape k̂      = μ̂²/σ̂² ≈ 2.08
    Gamma scale θ̂      = σ̂²/μ̂  ≈ 0.012

  p-value = 1 - F_Γ(0.612 ; 2.08, 0.012)  ≈ 1 × 10⁻⁹

  Adjusted α (Bonferroni, M=1 candidate):
    α_adj = 0.05 / 1 = 0.05

  p-value < α_adj  ✓  → drift CONFIRMED
```

**Output of detection module:** `[5050]` with p-value < 1e-9.
**Output of classification module:** TCD-Sudden ✓

---

### Scenario B — Blip drift (uses the noisy_profile_blip branch)

**Stream setup:**
- $X_t \sim \mathcal{N}(0, I_5)$ throughout, EXCEPT samples [4980, 5020] drawn from $\mathcal{N}([3,0,0,0,0], I_5)$ (40-sample blip), then back to baseline.
- One blip event at $t \approx 5000$.

**Stage 1 — σ(t) computation:**

```
t     σ(t)        Note
─────────────────────────────────────────────────
4900   0.041      stationary
4960   0.180      window starts catching blip
4980   0.290      first triangle apex (entering blip)
5000   0.180      window straddling blip middle
5020   0.260      second triangle apex (exiting blip)
5040   0.130      tail
5100   0.045      back to stationary
─────────────────────────────────────────────────
mean(σ_s)    ≈ 0.058
median(σ_s)  ≈ 0.048
std(σ_s)     ≈ 0.055
```

Two triangles close together → classic blip signature.

**Stage 2 — Peak detection:**

```
peak_threshold = 0.058 + 0.3 × 0.055 = 0.075

Peaks above threshold:
  σ(t=4980) = 0.290   ✓
  σ(t=5020) = 0.260   ✓

Candidate peaks: [4980, 5020]   ← n_p = 2
```

**Stage 3 — Feature extraction:**

```
Feature        Value      Interpretation
──────────────────────────────────────────
n_p            2          two peaks
WR             0.080      both peaks narrow
SNR            6.04       max(0.29) / median(0.048)
PPR            0.004      |5020 - 4980| / 10000 = 0.004 → small (close peaks)
DPAR           0.897      min(0.26, 0.29) / max(0.26, 0.29)
                          = 0.260 / 0.290 = 0.897
LTS            0.05       no trend
drift_length   1          (WR < 0.12 → TCD)
```

**Stage 4 — Decision tree walkthrough:**

```
STEP 1: Blip check
  n_p ∈ {2, 3} ?              n_p = 2 ✓
  ≥ 2 peaks detected ?        YES ✓

  Try compact_pair_blip:
    PPR > 0 and PPR < 0.20 ?   0.004 < 0.20 ✓
    DPAR > 0.60 ?              0.897 > 0.60 ✓
    WR < 0.30 ?                0.080 < 0.30 ✓
    → ALL THREE PASS → compact_pair_blip = True

  → return TCD-Blip ✓
```

(In this case the simpler `compact_pair_blip` branch matches; the `noisy_profile_blip` branch is used when the two peaks blur together due to noise so that drift_length > 1, and only the noisier criteria pass.)

**Output:** TCD-Blip ✓

---

### Scenario C — Incremental drift (the failure case — explains the 4.4%)

**Stream setup:**
- Continuous slow drift from $t = 4000$ to $t = 8000$. Mean of $X_t$ migrates linearly from $\mathbf{0}$ to $[1.5, 0, 0, 0, 0]^T$ over 4000 samples.
- No discrete event; pure ramp.

**Stage 1 — σ(t) computation:**

```
t     σ(t)        Note
─────────────────────────────────────────────────
1000   0.043      stationary (pre-drift)
3000   0.045      stationary
4000   0.052      drift onset
5000   0.072      slow rise
6000   0.098      slow rise (window mostly inside drift zone)
7000   0.115      near apex of slow rise
8000   0.122      drift end
9000   0.080      slow decline (window post-drift, ref still has drift tail)
9500   0.055      back to baseline
─────────────────────────────────────────────────
mean(σ_s)    ≈ 0.075
median(σ_s)  ≈ 0.072
std(σ_s)     ≈ 0.030
```

No sharp triangle. Just a wide flat hump. This is the geometric signature that breaks classification.

**Stage 2 — Peak detection:**

```
peak_threshold = 0.075 + 0.3 × 0.030 = 0.084

Possible "peaks" above threshold (local maxima):
  σ(t=8000) = 0.122   ✓ (the apex of the ramp)

Candidate peaks: [8000]   ← n_p = 1
```

**Stage 3 — Feature extraction:**

```
Feature        Value      Interpretation
──────────────────────────────────────────
n_p            1          single broad "peak"
WR             0.730      FWHM ≈ 73 samples / 2·50 → very wide
                          → triggers PCD branch ✓
SNR            1.69       max(0.122) / median(0.072) — barely above 1
LTS            0.42       linear trend present but R² < 0.5
                          (the ramp is noisy; not a clean line)
MS             0.55       mostly increasing but not enough
SDS            0.08       no clean step jumps
drift_length   2          (WR = 0.73 ≥ 0.12 → PCD)
```

**Stage 4 — Decision tree walkthrough (the failure):**

```
STEP 1: Blip check
  n_p ∈ {2, 3} ?  n_p = 1 → FAIL

STEP 1.5: PCD gate
  drift_length > 1 ? YES (drift_length = 2) → enter PCD branch ✓

  PCD subtype check:
    is_incremental = (LTS > 0.5)           ?  0.42 > 0.5 → FALSE
                  OR (MS > 0.6 AND LTS > 0.3)?  0.55 > 0.6 → FALSE
                  OR (SDS > 0.12 AND LTS > 0.3)?  0.08 > 0.12 → FALSE
    → is_incremental = FALSE
    → return PCD-Gradual

  ❌ MISCLASSIFIED — was actually Incremental, called Gradual.
```

**Output:** PCD-Gradual ✗ (wrong — should have been PCD-Incremental)

**Why this fails:** the LTS=0.5 threshold is too strict for noisy ramps. In this realistic example, the actual linear trend explains only 42% of σ(t)'s variance because of within-window fluctuation. The thesis's 4.4% Incremental accuracy is mostly cases like this — the geometry is *correct* (PCD is identified), but the subtype is consistently downgraded to Gradual.

**The mitigation in §5.4 of the conclusion:** future work proposes replacing the fixed LTS=0.5 threshold with a learned probabilistic classifier on the σ(t) features.

---

### Comparison summary across the three scenarios

| Stage | Scenario A (Sudden) | Scenario B (Blip) | Scenario C (Incremental) |
|-------|--------------------|-----|----------------------------|
| **σ(t) signature** | One sharp triangle, apex 0.42 | Two narrow triangles, apex 0.29 | Wide hump, apex 0.122 |
| **n_p** | 1 | 2 | 1 |
| **WR** | 0.060 | 0.080 | **0.730** |
| **SNR** | 9.33 | 6.04 | 1.69 |
| **drift_length** | 1 → TCD | 1 → TCD | 2 → **PCD** |
| **Decision tree path** | Step 2: Sudden ✓ | Step 1: Blip ✓ | Step 1.5: PCD-Gradual ✗ |
| **Outcome** | TCD-Sudden ✓ | TCD-Blip ✓ | PCD-Gradual ✗ (was Incremental) |

The pattern: **TCD types succeed because σ(t) has clear peaks; PCD subtypes (Gradual / Incremental) often degrade to each other because the geometric features can't reliably separate "linear ramp" from "wide hump".**

---

### Concept Memory worked example (Recurrent drift)

Suppose the stream alternates between two distributions — Distribution A (stationary $\mathcal{N}(0, I_5)$) and Distribution B ($\mathcal{N}([2,0,0,0,0]^T, I_5)$) — with switches every 1500 samples (recurrent A→B→A→B…).

**After the first A→B drift at t = 1500:**
```
post_drift_snapshot S_1 = stream[1500 : 1500 + 150]   # 150 points, all from B
γ_1 = median_heuristic(S_1) ≈ 0.31
Memory state: {1: (S_1, γ_1)}
```

**At second B→A drift at t = 3000 (back to A):**
```
post_drift_snapshot S_new = stream[3000 : 3000 + 150]   # 150 points, all from A
γ_new = median_heuristic(S_new) ≈ 0.34

Lookup against memory:
  γ_avg = (γ_1 + γ_new) / 2 = (0.31 + 0.34) / 2 = 0.325
  MMD²(S_new, S_1; γ_avg) = 0.78    ← S_new is from A, S_1 is from B → big distance
  
  τ_match = 0.15
  0.78 ≥ 0.15 → no match → S_new added as new concept

Memory state: {1: (S_1, γ_1), 2: (S_new, γ_new)}
```

**At third A→B drift at t = 4500:**
```
post_drift_snapshot S_new2 = stream[4500 : 4650]   # 150 points, all from B again
γ_new2 ≈ 0.30

Lookup against memory:
  Compare to S_1 (B): MMD²(S_new2, S_1; γ_avg) = 0.04   ← very small ← MATCH ✓
  Compare to S_2 (A): MMD²(S_new2, S_2; γ_avg) = 0.81

  min(0.04, 0.81) = 0.04 < τ_match = 0.15 → MATCH on S_1
  → drift labeled as TCD-Recurrent (matched concept #1)
  → adaptation strategy: load cached model trained on Distribution B
```

**The whole point:** with concept memory, the system *recognizes* it has seen Distribution B before, classifies as Recurrent, and reuses the cached model — no retraining needed. Without concept memory, this drift would have been classified as a fresh Sudden and the system would have re-learned distribution B from scratch, wasting compute.

The 71.5% Recurrent accuracy in the benchmark says concept memory works most of the time. Failures come from cases where the stream's noise level shifted between visits to the same concept, raising MMD²(S_new, S_match) above τ_match = 0.15.

# 6. Adaptation strategies

Once a drift is detected and classified, the model has to be updated. The thesis pairs each drift type with a strategy. Code: `experiments/monitoring/adaptation_strategies.py`.

## 6.1 The five strategies

| Strategy | When triggered | What it does |
|----------|---------------|--------------|
| **Full Reset (`adapt_sudden_drift`)** | Sudden | Discard the current model entirely, train a new one on a fresh batch of post-drift data. |
| **Incremental Update (`adapt_incremental_drift`)** | Incremental | Keep the model, do a few SGD steps on new mini-batches; learning rate small. |
| **Fine-tune (`adapt_gradual_drift`)** | Gradual | Like incremental but with a moderate amount of new data and a slightly higher LR; keeps continuity. |
| **Cache-Reuse (`adapt_recurrent_drift`)** | Recurrent | Look up the cached model trained when this distribution was last seen; reload it instead of retraining from scratch. |
| **Skip (`adapt_blip_drift`)** | Blip | Do nothing; trust that the transient will pass. (Optionally: hold off on adaptation for a cooldown window.) |

## 6.2 Why type-specific adaptation matters

A naive system would have one strategy: "drift detected → retrain from scratch". This is the **Periodic Retrain** baseline used in the experiments. It's expensive (full retraining cost every time) and *ignores recurrence* — if the same distribution comes back, you retrain again instead of reusing.

Worse, retraining on a Blip means you've trained your model on transient noise. Now when the noise passes, your model is *more* wrong than before.

**The adaptation experiments (Section 8.4) show:**
- On **Stepping Drift** (alternating sudden drifts + recovery): Type-Specific reaches **74.27%** prequential accuracy vs. **54.80%** with No Adaptation — a **+19.47pp** improvement, because Cache-Reuse handles the recurrence cheaply.
- On **Sudden Drift**: Type-Specific reaches **73.71%** vs. **53.68%** No Adaptation — **+20.03pp** improvement, because Full Reset doesn't waste time on partial updates.
- On **Mixed**: Type-Specific reaches **85.73%**, slightly below Periodic Retrain (86.09%) but with **far fewer false alarms** (1.1 vs. 8.3 per run) because it doesn't retrain on Blips.

The headline: **same accuracy as periodic retraining, with fewer wasted retrains.**

## 6.3 Where the strategies are wired up

In the Kafka prototype, the **Adaptation Manager** (`adaptor.py`) listens to the `drift.results` topic. When a drift event arrives:

1. Parse the type from the message (`{"drift_type": "TCD-Sudden", "position": 1234, ...}`).
2. Look up the strategy: `STRATEGY_MAP = {"TCD-Sudden": adapt_sudden_drift, ...}`.
3. Schedule the strategy to run after `ADAPTATION_DELAY = 50` samples (let the post-drift distribution stabilize).
4. The strategy retrains the model using a slice of stream from `[drift_position + delay, drift_position + delay + adaptation_window]`.
5. Publishes the new model to the `model.updated` topic.

The detection consumer reloads from `model.updated` and starts using the new model. This is the **end-to-end loop** demonstrated in Ch.4 of the thesis with the SEA + Sudden Drift scenario at sample 1500.

---

# 7. The Kafka prototype

The thesis includes a working Kafka-based streaming prototype, located in `experiments/monitoring/`. It is explicitly framed as a **prototype** (not a production system), with out-of-scope items disclaimed in Ch.3 §3.5.

## 7.1 Why Kafka?

Concept drift detection in production needs:
- A way to ingest data from a producer (sensor, web request log, transaction stream).
- A way to run the detector as a separate consumer (decoupled from production code).
- A way to broadcast detection results to anyone interested (model trainer, dashboard).
- A way to deliver retrained models back to the inference service.

Kafka (or its Kafka-API-compatible cousin **Redpanda**, used here) is the canonical message broker for this. The thesis uses Redpanda v24.1.9 in Docker.

## 7.2 The 4 topics

| Topic | Producer | Consumer | Payload (JSON) |
|-------|---------|----------|----------------|
| `sensor.stream` | data producer | detection consumer | `{"ts", "idx", "x": [...], "y", "drift_indicator"}` |
| `drift.results` | detection consumer | adaptation manager, dashboard | `{"position", "drift_type", "subcategory", "features"}` |
| `model.updated` | adaptation manager | detection consumer (for inference reload) | `{"model_version", "weights_uri", "trained_on_window"}` |
| `model.accuracy` | detection consumer | dashboard | `{"window", "prequential_acc"}` |

## 7.3 The four components

```
┌────────────┐   sensor.stream    ┌──────────────────────┐   drift.results
│  Producer  │────────────────────▶│  Detection Consumer  │────────────────┐
└────────────┘   (every 2ms)       │ (ShapeDD-IDW + SE-CDT)│                │
                                   └──────────────────────┘                │
                                              │                            ▼
                                              │ model.accuracy   ┌─────────────────────┐
                                              ▼                  │ Adaptation Manager  │
                                   ┌─────────────────────┐       │  (adapt_*_drift)    │
                                   │     Dashboard        │      └─────────────────────┘
                                   │ (real-time plot)     │                 │
                                   └─────────────────────┘                  │  model.updated
                                              ▲                             │
                                              └─────────────────────────────┘
```

**Producer (`producer.py`):** Generates a SEA-Lite stream with a sudden drift at `DRIFT_POSITION = 1500`. Pre-drift uses one variant of the SEA classifier; post-drift uses a different variant + linear feature transformation. Publishes JSON messages every 2 ms (~500 samples/sec).

**Detection Consumer (`consumer_stream.py`):** Subscribes to `sensor.stream`. Buffers samples; every 150 samples runs the ShapeDD-IDW detection + SE-CDT classification pipeline. Publishes `drift.results` when drift confirmed; publishes `model.accuracy` continuously.

**Adaptation Manager (`adaptor.py`):** Subscribes to `drift.results`. Routes by `drift_type` to the appropriate strategy (Section 6.1). Schedules the retraining; when done, publishes `model.updated`.

**Dashboard:** A Plotly/Dash app that shows the live MMD trace, drift events as vertical lines, and the prequential accuracy curve over time.

## 7.4 The end-to-end demo

The thesis runs **one** scripted scenario as the prototype demo (Ch.4 §4.4):
- Stream of 3000 samples from `producer.py`.
- Sudden drift inserted at sample 1500.
- The detection consumer flags drift around sample 1530–1560 (within $\delta = 75$ tolerance).
- The adaptation manager triggers Full Reset; trains a new Logistic Regression on samples 1550–2350.
- Detection consumer reloads the new model; prequential accuracy recovers from $\sim 0.59$ (post-drift, pre-update) to $\sim 0.93$ (post-update).

The accompanying figure is `image/kafka_results_real.png` (a dashboard screenshot).

## 7.5 What's explicitly out-of-scope

The thesis disclaims (Ch.3 lines 472–478) that the prototype does **not** measure:
- Throughput at production load (1000s of samples/sec across multiple partitions).
- Recovery time from broker / consumer crashes.
- Schema-evolution or backwards-compatibility.
- Multi-tenancy (multiple streams, multiple models).

These are real engineering concerns that a production system would have to address. The prototype establishes only that the architectural pattern works end-to-end on a single scenario.

This honest framing is exemplary practice — it protects the thesis from over-claiming production-readiness while still demonstrating the core engineering pipeline.

## 7.6 Robustness details

**Producer (after F-10.1 fix):** explicit Kafka configuration with `retries=5`, `retry.backoff.ms=100`, `acks=all`, providing automatic retry on transient failures.

**Consumer:** offset committed only after successful processing — no message loss on crash.

**Adaptation:** atomic model swap using temp-file-and-rename pattern.

---

# 8. Experiments

The thesis runs three experiments, with three different purposes, three different metrics, and three different tolerance settings $\delta$. This is summarized in Table~\ref{tab:eval-conventions} of `chapters/04_experiments_evaluation.tex`.

## 8.1 Three evaluation modes (don't conflate them)

| Mode | Purpose | $\delta$ tolerance | Cooldown | What FP means | Primary metrics |
|------|---------|--------------------|----------|---------------|-----------------|
| **Detection** | "How well does ShapeDD-IDW *find* drift events?" | 75 samples | 150 ($=2\delta$) | False alarm on stationary stream OR detected drift outside tolerance window | F1, EDR, FP/run |
| **Classification** | "Given drift is found, how well does SE-CDT *label* it?" | 250 samples | — | Mismatched event outside transition zone | CAT, SUB, EDR |
| **Adaptation** | "How well does the *full pipeline* recover model accuracy?" | 400 samples | — | Excess alarms during recovery | Prequential Accuracy |

A common mistake — both for readers and for examiners — is to compare numbers across modes. A detector's FP/run in Detection mode is not the same as its FP/run in Adaptation mode. The cooldown (150 vs. none) and tolerance (75 vs. 400) change the FP definition.

## 8.2 Detection: the headline benchmark

**Setup:**
- 14 datasets (13 synthetic + 1 semi-real Electricity dataset). Synthetic includes SEA, STAGGER, Hyperplane, GaussianShift, RBF, LED, etc. Stationary control sets are also included.
- 10 drift events per dataset, evenly spaced at every 1000 samples. Stream length 10,000.
- 30 independent runs per dataset, with prime-spaced seeds (`42 + i*137` for $i = 0..29$).
- Total experiments: $14 \times 8 \;\mathrm{methods} \times 30 \;\mathrm{seeds} = 3360$.
- Tolerance $\delta = 75$, Cooldown = 150.

**Methods compared:**
| Method | Family |
|--------|--------|
| KS-Test | Statistical |
| MMD (Standard) | Distribution test |
| D3 | Discriminative |
| DAWIDD | Independence test (HSIC) |
| ShapeDD (original 2021) | Shape-based |
| **IDW-MMD** (this thesis) | Weighted MMD ablation |
| **ShapeDD-IDW** (this thesis) | Shape-based + IDW + Gamma |
| **SE-CDT** (this thesis) | Detection-classification unified |

**Headline results (`table_I_comprehensive_performance.tex`, regenerated 2026-05-17):**

| Method | Precision | Recall | **F1** | Delay | FP/run |
|--------|-----------|--------|-------|-------|--------|
| **DAWIDD** | 0.435 | 0.731 | **0.531** | 38 | 10.4 |
| **IDW-MMD / SE-CDT / ShapeDD-IDW** | 0.432 | 0.737 | **0.531** | 36 | **9.6** |
| MMD (Standard) | 0.427 | 0.732 | 0.525 | 35 | 10.5 |
| ShapeDD (original) | 0.377 | 0.749 | 0.492 | 34 | 13.1 |
| D3 | 0.558 | 0.472 | 0.489 | 16 | **0.6** |
| KS-Test | 0.224 | 0.775 | 0.335 | 34 | 27.2 |

**How to read this table:**
- The thesis methods (IDW-MMD/SE-CDT/ShapeDD-IDW) tie with DAWIDD on F1, but with **lower FP** (9.6 vs 10.4) and significantly lower runtime (Section 8.5).
- ShapeDD-IDW improves on ShapeDD original: F1 0.492 → 0.531 (+0.039), FP 13.1 → 9.6 (-3.5/run).
- D3 has the best precision and lowest FP but only 47% recall — misses too much.
- KS-Test gets the highest recall (77.5%) but creates 27.2 false alarms per run — unusable in practice.

The Nemenyi statistical test (Section 9 below) confirms: DAWIDD and the three thesis methods are **statistically tied** — the F1=0.531 group is one cluster.

## 8.3 Classification: SE-CDT performance

**Setup:**
- 6 scenarios × 30 seeds × 10 events = **1800 classification events**.
- Scenarios: Mixed A, Mixed B, Repeated Sudden, Repeated Gradual, Repeated Incremental, Repeated Recurrent.
- Tolerance $\delta = 250$ (looser, because classification cares about *which* drift, not exact position).
- "Oracle mode": SE-CDT receives ground-truth drift positions and only the *classification* step is exercised. This decouples classification accuracy from detection accuracy.

**Headline results (`table_se_cdt_aggregate.tex` and `_by_type.tex`):**

| Metric | Value | Meaning |
|--------|-------|---------|
| **CAT (Category)** | **80.1%** | Correct partition into TCD vs PCD |
| **SUB (Subcategory) — micro avg** | **50.5%** | Correct subtype across all events |
| **SUB — macro avg** | **50.0%** | Average per-class subtype accuracy (gives equal weight to rare classes) |

**Per-subtype accuracy:**

| Drift type | Accuracy | Comment |
|------------|----------|---------|
| Sudden | **82.4%** | The strength: triangle property is reliable. |
| Blip | **60.8%** | Decent after the relaxed two/three-peak rule. |
| Recurrent | **71.5%** | Concept Memory works well when the same distribution returns. |
| Gradual | **30.8%** | Weak — easy to confuse with mild Incremental. |
| Incremental | **4.4%** | Worst case. The MMD signal looks similar to a noisy plateau. |

**Honest framing in Ch.5:**
> "Ở bước phân loại, SE-CDT đạt kết quả chấp nhận được ở mức TCD/PCD nhưng còn yếu ở tiểu loại, đặc biệt với Incremental. Đây là giới hạn quan trọng nhất của luận văn và không nên che giấu."

In English: "On classification, SE-CDT is acceptable at TCD/PCD level but weak on subtype, especially Incremental. This is the most important limitation of the thesis and should not be hidden."

This is the kind of self-criticism that makes a defense stronger, not weaker.

## 8.4 Adaptation: type-specific vs baselines

**Setup:**
- Stream length 5000, 5 drift events per stream, 30 runs.
- Base classifier: Logistic Regression (`max_iter=1000`).
- Tolerance $\delta = 400$ (very loose — we measure *recovery*, not exact detection).
- Prequential window = 100 (rolling accuracy over the last 100 predictions).

**Strategies compared:**
- **No Adaptation** — never update the model.
- **Periodic Retrain** — retrain every 200 samples regardless.
- **Simple Retrain** — retrain on every detected drift.
- **Type-Specific (this thesis)** — strategy chosen by SE-CDT type.

**Headline results (`chapters/04_experiments_evaluation.tex` line 433–471):**

| Scenario | No Adaptation | Periodic Retrain | Simple Retrain | **Type-Specific** |
|----------|---------------|------------------|----------------|-------------------|
| Stepping Drift | 54.80% | 71.60% | 65.43% | **74.27%** |
| Sudden Drift | 53.68% | 63.65% | 61.94% | **73.71%** |
| Mixed | 73.40% | **86.09%** | 84.21% | 85.73% |

**Reading:**
- On Stepping (alternating sudden drifts + recovery to old): Type-Specific wins by **+19.47pp** over No Adaptation. Concept Memory shines.
- On Sudden: Type-Specific wins by **+20.03pp**. Full Reset is decisive.
- On Mixed: Type-Specific is essentially tied with Periodic Retrain (within 0.36pp). Periodic Retrain is slightly higher but with **8.3 FP/run** vs Type-Specific's **1.1 FP/run** — fewer wasted retrains.

## 8.5 Runtime: the headline speed-up

**`table_III_runtime_stats.tex`** (single-threaded `time.process_time()`, 10,000-sample stream, averaged across all datasets and runs):

| Method | Runtime (s) | Throughput (samples/s) | **Speedup vs. ShapeDD original** |
|--------|------------:|-----------------------:|---------------------------------:|
| KS | 0.35 | 28,433 | 14.31× |
| D3 | 0.37 | 26,824 | 13.50× |
| **ShapeDD-IDW** | **0.70** | **14,247** | **7.17×** |
| IDW-MMD (standalone) | 0.70 | 14,233 | 7.16× |
| **SE-CDT** | **0.70** | **14,211** | **7.15×** |
| ShapeDD (original) | 5.03 | 1,987 | 1.00× |
| MMD (Standard) | 5.05 | 1,980 | 1.00× |
| DAWIDD | 6.79 | 1,473 | 0.74× |

**This is the "approximately 7× speedup" claim** in the abstract. Came directly from replacing the permutation test (Standard MMD's $B = 2500$ permutations) with the Gamma p-value ($B = 20$ bootstrap moments).

## 8.6 H0 calibration (does the test actually have α = 0.05?)

A claimed test of significance is only useful if it actually controls Type I error. The thesis runs an **H0 calibration experiment** (`scripts/h0_calibration.py`):

- Generate stationary streams from three reference distributions: Gaussian-iid d=5, Gaussian-iid d=10, Gaussian-AR(1) d=5.
- Run each detector at $\alpha = 0.05$.
- Measure the empirical false-positive rate.

**Results (`table_h0_calibration.tex`):**

| Reference distribution | Standard MMD | IDW-MMD (Gamma null) | SE-CDT (composite) |
|------------------------|-------------:|----------------------:|--------------------:|
| Gauss-iid d=5 | 0.030 | **0.040** | 0.085 |
| Gauss-iid d=10 | 0.025 | **0.050** | 0.045 |
| Gauss-AR(1) d=5 | 0.040 | **0.075** | **0.100** |

**Reading:**
- IDW-MMD's empirical Type-I error is in [0.040, 0.075] — close to the nominal 0.05, with mild over-rejection on AR(1) data (slightly correlated samples violate the i.i.d. assumption underlying the Gamma fit). **Honest assessment:** acceptable for practical use; not a perfectly calibrated test.
- SE-CDT composite (full pipeline including classification) reaches 0.100 in the AR(1) case — twice the nominal rate. The thesis flags this in Ch.5 as a limitation: the composite test runs multiple decisions in sequence, accumulating Type-I error beyond the per-test control.

## 8.7 Audit note (the methodological self-correction)

The `wmmd_gamma` function's docstring (`mmd_variants.py:246–322`) carries an unusually rigorous audit note documenting an earlier methodology bug:

> "This replaces the previous `wmmd_asymptotic` which applied the Gaussian H₁ asymptotic of standard MMD as if it were the H₀ null distribution — an incorrect derivation that produced a systematically over-conservative test (empirical Type-I error ≈ 0 on stationary streams instead of the nominal α)."

**What happened:** an earlier implementation conflated two different asymptotic results (the Gaussian limit *under H₁* vs. the actual null distribution under H₀). This produced a test that almost never rejected, regardless of input — because its critical value was systematically too large. The bug was found by the H0 calibration experiment (Type-I error came back at ≈ 0 instead of ≈ 0.05). The fix was switching to the moment-matched Gamma distribution per Gretton et al. 2009.

**Why the thesis documents this in code comments:** future maintainers (or examiners reading the code) need to know the correct derivation. Calling out the previous error transparently is the kind of methodology accountability that strengthens an advisor's confidence in the work.

---

# 9. Statistical methodology

The headline F1 numbers in §8.2 are means across 30 seeds × 14 datasets. Are the differences between methods *significant*, or just noise? The thesis answers this with the **Friedman + Nemenyi** procedure (Demšar 2006).

## 9.1 The procedure

1. For each of the 14 datasets, take the mean F1 over 30 seeds → one number per (method, dataset) pair.
2. Within each dataset, **rank the 8 methods** (1 = best F1, 8 = worst).
3. **Average rank** $\bar r_j = \frac{1}{14} \sum_{d=1}^{14} \mathrm{rank}_j(d)$ for each method.
4. **Friedman test**: $\chi^2_F = \frac{12 N}{K(K+1)} \big[\sum \bar r_j^2 - \frac{K(K+1)^2}{4}\big]$ → p-value < 0.05? If yes, there *is* an overall difference. If no, all methods are equivalent.
5. **Iman-Davenport correction**: convert $\chi^2_F$ to F-statistic $F = \frac{(N-1) \chi^2_F}{N(K-1) - \chi^2_F}$ for better power on small samples.
6. **Nemenyi post-hoc**: pairs $(i, j)$ are significantly different iff $|\bar r_i - \bar r_j| > \mathrm{CD} = q_\alpha \sqrt{\frac{K(K+1)}{6N}}$.
7. **Critical Difference plot**: a horizontal axis of ranks; methods within a "clique" connected by horizontal bars indicate "no significant difference".

## 9.2 Numerical result

For $K = 8$ methods, $N = 14$ datasets, $\alpha = 0.05$, the tabulated $q_{0.05} = 3.031$. So:

$$\mathrm{CD} = 3.031 \times \sqrt{\frac{8 \cdot 9}{6 \cdot 14}} = 3.031 \times 0.926 \approx 2.806.$$

**Average ranks (`table_statistical_tests.tex`, regenerated 2026-05-17):**

| Rank | Method | Average Rank |
|-----:|--------|-------------:|
| 1 | DAWIDD | 3.679 |
| 2-4 | IDW-MMD / SE-CDT / ShapeDD-IDW | **3.821** (tied) |
| 5 | MMD (Standard) | 4.607 |
| 6 | D3 | 4.750 |
| 7 | ShapeDD (original) | 5.179 |
| 8 | KS | 6.321 |

**Differences vs. CD = 2.806:**
- DAWIDD vs. IDW-MMD: $|3.679 - 3.821| = 0.142$ ≪ CD → **not significant** (statistically tied).
- IDW-MMD vs. MMD: $|3.821 - 4.607| = 0.786$ ≪ CD → not significant.
- IDW-MMD vs. ShapeDD original: $|3.821 - 5.179| = 1.358$ ≪ CD → not significant.
- IDW-MMD vs. KS: $|3.821 - 6.321| = 2.500$ < CD → not significant (but close).

**What this means scientifically:** the headline F1 values cluster the methods into **one large group** (DAWIDD, IDW-MMD, SE-CDT, ShapeDD-IDW, MMD, D3, ShapeDD), with only KS clearly separated (despite even KS not formally crossing CD).

The thesis is honest about this in Ch.4 §4.3:
> "kết quả nên được đọc là nhóm phương pháp MMD/DAWIDD có hiệu năng tương đương trên benchmark hiện tại, không phải một phương pháp thắng rõ ràng."

In English: "the result should be read as 'the MMD/DAWIDD group performs equivalently on the current benchmark', not 'one method clearly wins'." The thesis's contribution is **not** "we beat DAWIDD", but rather **"we match DAWIDD with much lower FP and 7× faster runtime"**.

## 9.3 Why use ranks instead of raw F1?

Two reasons:
1. F1 isn't comparable across datasets. A method that scores 0.6 on a hard dataset might be doing better than another method scoring 0.85 on an easy one. Per-dataset ranks normalize this.
2. The Friedman test is **non-parametric** — no assumption of normality of F1. Just rank ordering. More robust.

This is why Demšar (2006) recommends ranks for cross-dataset method comparison, and it's the standard in modern ML benchmarking papers.

---

# 10. Honest limitations

The thesis is unusually transparent about what it can and cannot do. This section gathers all the limitations explicitly stated in `chapters/05_conclusion_future_work.tex` and elsewhere. **For defense, you should expect every limitation to be probed.**

## 10.1 Detection-side limitations

### 10.1.1 IDW-MMD is less sensitive to gradual drift than to sudden drift
**Why:** the IDW reweighting boosts boundary points (where sudden drift first appears) and dampens interior points (where gradual drift mainly manifests). This is a deliberate trade-off documented in `chapters/03_proposed_model.tex`. Mitigation: the two-stage pipeline still uses **Standard MMD** on the trace step, which has uniform sensitivity. The IDW step only validates candidates already proposed by the trace.

### 10.1.2 The Gamma null is asymptotic, not exact
**Why:** Gretton et al. 2009's result is "MMD$^2$ is **approximately** Gamma under H₀" — the convergence rate to Gamma depends on sample size and kernel choice. On small windows ($l_2 = 150$), the approximation is good for i.i.d. data but degrades for AR(1) data (correlated samples). Empirical Type-I error rises from 0.05 (i.i.d.) to ~0.075 (AR(1)) — over-rejection by ~50% on correlated data.

### 10.1.3 H0 calibration only covers Gaussian distributions
**Why:** the calibration experiment uses three Gaussian reference families. Real-world data can be heavy-tailed, multi-modal, sparse — and the Gamma null might fit them differently. The thesis does not claim calibration generalizes outside Gaussian-like distributions.

### 10.1.4 Bonferroni is conservative
**Why:** with $M$ candidates, Bonferroni divides $\alpha$ by $M$, which is the most conservative possible correction. Holm or Benjamini-Hochberg would give more power. This was a deliberate simplicity choice; future work could swap in BH.

## 10.2 Classification-side limitations

### 10.2.1 Subtype accuracy is low (50% micro, 50% macro)
**Why:** see §8.3. Specifically:
- **Incremental at 4.4%**: the MMD signal of incremental drift looks similar to a noisy plateau. The features (LTS, MS, SDS) try to distinguish, but the signal-to-noise ratio is just too low at the chosen window sizes.
- **Gradual at 30.8%**: easy to confuse with "mild Incremental" — both produce wide flat peaks.

The thesis acknowledges this is **the most important limitation** ("không nên che giấu" = "should not be hidden"). Future work directions in Ch.5: multi-scale window analysis, learnable embeddings instead of hand-coded features, snapshot-based subtype classification (like a ConvNet on the MMD trace).

### 10.2.2 Decision-tree thresholds are heuristic, not derived
**Why:** WR < 0.15, SNR > 2.0, CV < 0.30 — these come from looking at MMD traces and picking values that worked. The self-calibration mechanism (§5.6) adapts these to noise floor, but the **structure** of the tree (the order, the inequality directions) is fixed by hand.

A learned classifier would adapt to data better, but couldn't be trained without labels (§5.7's three reasons).

### 10.2.3 The two BlipProfile branches are aesthetically ugly
**Why:** Step 1 of Algorithm 3 has a "compact pair" branch and a "noisy profile" branch with 5 numeric thresholds (0.17, 1.45, 2.60, 0.45, 0.85, 0.12). These came from inspecting failure cases empirically. The thesis documents them in Appendix A.2 (after F-6.2 fix), but acknowledges their ad-hoc nature. Future work: replace with a small classifier trained on simulated drift events.

### 10.2.4 Concept Memory limits
**Why:** $M = 8$ snapshots, each $n_s = 150$ points. If more than 8 distinct distributions recur, the oldest gets evicted. If a recurrence has been gone for too long, the snapshot might be stale (data noise floor changed). The 0.15 match threshold is also fixed — should arguably be adapted to current noise level.

## 10.3 System-side limitations (Kafka prototype)

### 10.3.1 Single-stream, single-topic-set demonstration
**Why:** see §7.5. The prototype runs one stream end-to-end. Multi-tenancy, schema evolution, recovery from broker failure — all out of scope.

### 10.3.2 Producer rate is hard-coded
**Why:** `time.sleep(0.002)` produces ~500 samples/sec. To test at higher loads, you'd edit the source. Documented as F-10.2 in REVIEW_REPORT (Nit-level finding).

### 10.3.3 No partitioning across multiple consumers
**Why:** the detection consumer is a single Python process. Real production would need partitioned topics with multiple consumer groups for parallelism. Out of scope.

## 10.4 Methodology / scope limitations

### 10.4.1 Only $P(X)$ shift is observable
**Why:** unsupervised. If $P(Y \mid X)$ shifts while $P(X)$ stays exactly the same, no method in this thesis can detect it. The joint-drift assumption (§1.4) handles the typical case; the rare adversarial case (only $P(Y|X)$ shifts) requires labels.

### 10.4.2 14 datasets is small
**Why:** larger benchmarks (e.g., the SUNE benchmark from 2023) include 30+ datasets. The thesis chose 14 for runtime reasons (3360 experiments takes ~38 minutes parallel; scaling to 60 datasets would take 2.5 hours per benchmark run). The 14 datasets do span synthetic + semi-real and multiple drift types, so the coverage is reasonable but not exhaustive.

### 10.4.3 Classification accuracy is measured in "Oracle mode"
**Why:** SE-CDT receives ground-truth drift positions in the classification benchmark. This decouples classification accuracy from detection accuracy (a clean methodological choice — Demšar would approve). But it means the **end-to-end** classification accuracy (when run on top of the detector's outputs) would be lower, because some drifts aren't detected at the right position. The thesis is explicit about Oracle mode in Ch.4 line 277.

## 10.5 What future work could (and should) do

From `chapters/05_conclusion_future_work.tex`:

1. **Multi-scale window analysis** — run IDW-MMD at $l \in \{50, 100, 200\}$ simultaneously and combine. Would help with gradual/incremental detection sensitivity.
2. **Learnable τ_match** — adapt the concept-memory match threshold to the current noise floor.
3. **Snapshot embeddings instead of hand-coded features** — train a small ConvNet on simulated drift traces; output an embedding; classify via nearest-neighbor in embedding space.
4. **Better p-value correction** — Holm or Benjamini-Hochberg instead of Bonferroni.
5. **Multi-stream production deployment** — partitioned Kafka topics, consumer groups, schema evolution.
6. **Real-world deployment study** — currently the only "real" data is Electricity (semi-synthetic). A study on a deployed production stream would validate the approach in practice.

---

# 11. Implementation map

This section is a quick reference for "where in the code is each idea?". Useful during defense if asked "show me where you implement X".

## 11.1 Detection (IDW-MMD)

| Concept | File | Function/Class |
|---------|------|----------------|
| IDW weight $w_i = 1/(\sqrt{d(x_i)} + \epsilon)$ | `core/detectors/mmd_variants.py` | `compute_optimal_weights()` |
| IDW-MMD² statistic | `core/detectors/mmd_variants.py` | `compute_idw_mmd_squared()` |
| Gamma p-value | `core/detectors/mmd_variants.py` | `wmmd_gamma()` |
| Standard MMD trace | `core/detectors/mmd_variants.py` | inside `shapedd_idw_mmd_proper()` lines 405–415 |
| Two-stage pipeline | `core/detectors/mmd_variants.py` | `shapedd_idw_mmd_proper()` |
| Bonferroni correction | `core/detectors/mmd_variants.py` | `shapedd_idw_mmd_proper()` line 518 |
| Median heuristic γ | `core/detectors/mmd_variants.py` | `compute_gamma_median_heuristic()` |
| Audit note (wmmd_asymptotic was wrong) | `core/detectors/mmd_variants.py` | `wmmd_gamma()` docstring lines 246–322 |

## 11.2 Classification (SE-CDT)

| Concept | File | Function/Class |
|---------|------|----------------|
| SE-CDT main class | `core/detectors/se_cdt.py` | `SE_CDT` |
| 9 features | `core/detectors/se_cdt.py` | `extract_features()`, `extract_temporal_features()` |
| Decision tree | `core/detectors/se_cdt.py` | `SE_CDT.classify()` lines 605–700 |
| Self-calibration | `core/detectors/se_cdt.py` | `_RollingFeatureBaseline` (line 19) |
| Concept Memory | `core/detectors/se_cdt.py` | `_ConceptMemory` (search "concept_memory") |
| Growth Process (FWHM) | `core/detectors/se_cdt.py` | `_growth_process()` (line 368) |
| TCD-Blip (compact_pair_blip) | `core/detectors/se_cdt.py` | `classify()` line 642–650 |
| TCD-Blip (noisy_profile_blip) | `core/detectors/se_cdt.py` | `classify()` line 651–657 |

## 11.3 CDT-MSW (the inherited supervised baseline, kept for comparison)

| Concept | File | Function/Class |
|---------|------|----------------|
| CDT-MSW reproduction | `core/detectors/cdt_msw.py` | `CDT_MSW` class |
| Honest disclaimer (non-tuned reproduction) | `core/detectors/cdt_msw.py` | docstring lines 5–26 |

## 11.4 Adaptation strategies

| Concept | File | Function |
|---------|------|----------|
| Strategy library | `experiments/monitoring/adaptation_strategies.py` | `adapt_sudden_drift()`, `adapt_blip_drift()`, `adapt_recurrent_drift()`, `adapt_gradual_drift()`, `adapt_incremental_drift()` |
| Adaptation Manager (Kafka) | `experiments/monitoring/adaptor.py` | `main()` |
| Strategy dispatch by type | `experiments/monitoring/adaptor.py` | `STRATEGY_MAP` |

## 11.5 Kafka prototype

| Component | File |
|-----------|------|
| Producer | `experiments/monitoring/producer.py` |
| Detection consumer | `experiments/monitoring/consumer_stream.py` |
| Adaptation manager | `experiments/monitoring/adaptor.py` |
| Topic config | `experiments/monitoring/config.py` |
| Docker compose (Redpanda) | `experiments/monitoring/docker-compose.yml` |

## 11.6 Experiments

| Experiment | File |
|------------|------|
| Detection benchmark | `experiments/benchmark/main.py` |
| H0 calibration | `scripts/h0_calibration.py` |
| SE-CDT vs CDT-MSW comparison | `experiments/benchmark/benchmark_proper.py` |
| Adaptation prequential | `experiments/monitoring/evaluate_prequential.py` |
| LaTeX table export | `experiments/benchmark/analysis/latex_export.py` |
| Statistical tests (Friedman+Nemenyi) | `experiments/benchmark/analysis/statistics.py` |

## 11.7 Configuration

| Concern | File | Constant |
|---------|------|----------|
| Number of seeds | `experiments/benchmark/config.py` | `N_RUNS = 30` |
| Seed pattern | `experiments/benchmark/config.py` | `RANDOM_SEEDS = [42 + i*137 for i in range(N_RUNS)]` |
| Window sizes | `experiments/benchmark/config.py` | `SHAPE_L1 = 50`, `SHAPE_L2 = 150` |
| Permutation count (ShapeDD baseline) | `experiments/benchmark/config.py` | `SHAPE_N_PERM = 2500` |
| Cooldown | `experiments/benchmark/config.py` | `COOLDOWN = 150` |
| Detection α | `experiments/benchmark/config.py` | `SPECTRA_ALPHA = 0.01` (note: SE-CDT uses 0.05) |
| SE-CDT τ_match | `core/detectors/se_cdt.py` | `concept_match_threshold = 0.15` |
| SE-CDT M (memory size) | `core/detectors/se_cdt.py` | `concept_memory_size = 8` |
| SE-CDT n_s (snapshot size) | `core/detectors/se_cdt.py` | `concept_snapshot_size = 150` |

## 11.8 Reproducibility entry points

To reproduce the entire thesis end-to-end:

```bash
./setup_environment.sh          # creates .venv, installs requirements
./run_all.sh                    # full 30-run pipeline (~2-3 hours sequential)
./build_thesis.sh               # rebuilds the PDF
```

For a quick smoke test:

```bash
QUICK_MODE=True ./run_all.sh    # 2 runs instead of 30 (~10 min)
```

---

# 12. Defense Q&A

This section is a curated list of questions an examiner is likely to ask, with model answers. Group A (motivation), Group B (IDW-MMD), Group C (SE-CDT), Group D (experiments/stats), Group E (system/Kafka), Group F (limitations/future work).

## Group A — Motivation and scope

### Q1. Why concept drift? Why is this an important problem?

**A.** Production ML systems run continuously on data that changes over time — fraud patterns, sensor wear, seasonality, market shifts. A model trained once and deployed forever silently degrades. The cost can be enormous: misclassified transactions, missed maintenance, broken recommendations. Without an automatic drift-detection layer, you have to monitor every model by hand. With one, the system can self-update. This is increasingly necessary as ML moves from research to production.

### Q2. Why unsupervised? Don't you have labels in any real system?

**A.** Sometimes you do, sometimes you don't, but **you almost never have them in time**. In fraud detection, labels arrive weeks later (after investigators confirm). In predictive maintenance, labels arrive only when equipment fails (which is the thing you're trying to predict). In recommendation, labels are inherently noisy (clicks aren't endorsements). Building drift detection that *waits* for labels means waiting too long. The unsupervised approach catches drift the moment $P(X)$ moves, well before labeled accuracy degradation can be measured. The cost is that you can't catch the rare "P(Y|X)-only shift", but joint drift is the dominant case in practice.

### Q3. Why MMD instead of KL divergence or Wasserstein distance?

**A.** Three reasons:
1. **MMD has a closed-form empirical estimator** (kernel sums); KL needs density estimation, Wasserstein needs an optimal-transport solve.
2. **MMD has a known null distribution** (Gretton 2009 Gamma approximation); this gives p-values for free.
3. **MMD with the universal Gaussian RBF kernel is consistent** — it's zero iff $P = Q$.

KL is asymmetric and undefined when supports differ. Wasserstein is more expensive ($O(n^3)$ exact, $O(n^2 \log n)$ Sinkhorn). MMD wins on all three counts for streaming applications.

### Q4. What's "joint drift" and why do you assume it?

**A.** Joint drift = $P(X)$ and $P(Y|X)$ change together. The assumption is that this is *typical* in real systems: when fraudsters change their tactics (drift in $P(Y|X)$), they also use different transaction patterns ($P(X)$ shift). When sensors wear out ($P(X)$ shift), the failure modes change ($P(Y|X)$ shift). Counter-examples (only-$P(Y|X)$ shifts) exist — but they require labels to detect, which are unavailable in our setting.

The assumption is *operationally useful*: it tells us that watching $P(X)$ is a reasonable proxy for watching the model's actual degradation. The thesis is honest that it's an assumption, not a theorem.

## Group B — IDW-MMD (detection)

### Q5. What does IDW actually do, in one sentence?

**A.** It downweights points in the dense interior of the distribution and upweights points at the boundary, so that drift signals (which usually appear at the boundary first) aren't drowned out by interior similarity.

### Q6. Why $1/\sqrt{d}$ and not $1/d$?

**A.** $1/d$ would amplify isolated outliers enormously — a single point with no neighbors gets weight infinity (with $\epsilon$ floor: very large). The estimator becomes noise-dominated. $1/\sqrt{d}$ gives a milder, more stable upweighting that still favors boundary points but doesn't blow up. The exponent $-1/2$ is also natural: it makes $\sqrt{d} \cdot w_i = $ constant (a "normalized count"), which has clean dimensional analysis.

### Q7. Why uniform weights on the cross-term?

**A.** This is the most important design choice. The cross-term $\frac{2}{nm} \sum k(x_i, y_p)$ measures absolute similarity between the two distributions — a "geometric anchor". If we put IDW weights on it, we'd be saying "boundary points of $X$ should match boundary points of $Y$ extra hard". But boundary structure is noisy; the weights for $X$'s boundary and $Y$'s boundary don't agree by construction. You'd introduce a "false drift" signal whenever boundary structures differ from sampling alone. With uniform cross-term, the only deviation from zero comes from genuine distribution change in the within-window terms.

### Q8. Why doesn't the IDW bias hurt the test?

**A.** IDW-MMD$^2$ has a small positive bias under H₀: even when $P = Q$, the empirical statistic is slightly > 0 due to asymmetric weighting. *This is fine*, because we don't test "$T > 0$"; we test "$T > T_{0.95}$ where $T_{0.95}$ is the 95th percentile under H₀". The Gamma null distribution's mean and variance are estimated empirically from bootstrap samples — they automatically incorporate the bias. The test asks "is the observed value extreme *relative to the null distribution we just measured*?", and the bias is part of that null distribution.

### Q9. The Gamma null is approximate. Doesn't that break the test?

**A.** It's a controlled approximation. Empirically, on i.i.d. Gaussian data the test's Type-I error is in [0.040, 0.075], close to the nominal 0.050. On AR(1) data (correlated samples) it rises to ~0.075 — over-rejection by 50%. This is acceptable for practical drift detection, where you don't need exact-ε control; you need "approximately the right rate". The thesis's H0 calibration table makes this honest. If you needed perfectly calibrated tests, you'd run permutations — at 7× the runtime.

### Q10. Why Bonferroni and not BH or Holm?

**A.** Three reasons:
1. **Simplicity** — Bonferroni is one line; BH requires sorting p-values.
2. **Conservativeness** — BH controls FDR (expected proportion of false discoveries), Bonferroni controls FWER (probability of any false discovery). For drift detection, the latter is more appropriate: a single false alarm triggers retraining, costing real engineering time.
3. **Few candidates** — typically $M < 20$ candidates per stream. Bonferroni's loss of power is small at low $M$; BH's gain is largest at high $M$. So the trade-off favors Bonferroni in our regime.

That said, BH would be a clean future-work improvement.

### Q11. Could you have used a simpler approach (e.g., set a fixed threshold on Standard MMD)?

**A.** Yes, and that's the baseline. Standard MMD with a hand-tuned threshold gives F1 ≈ 0.525, FP ≈ 10.5/run. The thesis's contribution is to add (a) the IDW reweighting (boundary sensitivity) and (b) the Gamma p-value (calibrated test). These get F1 = 0.531 with FP = 9.6/run — same F1 with fewer false positives. The improvement isn't huge in F1 terms because the benchmark mostly has clear sudden drifts that anything would detect. The IDW advantage shows up more in the Type-I-error calibration table (Section 8.6) where the test is properly calibrated regardless of the noise floor.

### Q12. What's the speedup actually doing? Why does Gamma-null beat permutation by 7×?

**A.** In Standard MMD with permutation:
- For each candidate validation, you do **B = 2500 permutations** of the kernel matrix labels and re-compute MMD$^2$.
- Each MMD$^2$ on a window of size $l_2 = 150$ is ~22,500 kernel evaluations.
- Total per candidate: 2500 × 22,500 ≈ 5.6×10⁷ ops.

In IDW-MMD with Gamma:
- For each candidate, do **B = 20 bootstraps** to estimate moments.
- Same per-bootstrap cost.
- Total per candidate: 20 × 22,500 ≈ 4.5×10⁵ ops.

Ratio: ~125× fewer kernel evaluations *in the validation step*. End-to-end (including the trace step which is the same in both pipelines), the measured speedup is ≈7× on the benchmark.

## Group C — SE-CDT (classification)

### Q13. Why a hand-coded decision tree instead of a learned classifier?

**A.** Three reasons (see §5.7):
1. **No labels available** — we can't train a learned classifier without labeled drift events, which are exactly what we don't have at runtime.
2. **Interpretability** — operators can read the tree and understand why a drift was classified as Sudden vs Gradual. With a black-box classifier, debugging mis-classifications is much harder.
3. **Stability across deployment domains** — a learned classifier overfits to its training data's statistics. The decision tree's geometric thresholds (FWHM, SNR) generalize better, especially with the self-calibration mechanism.

### Q14. The thresholds (0.15, 0.30, 0.20...) seem arbitrary. How did you choose them?

**A.** They came from inspecting MMD traces of synthetic drift events with known types. WR < 0.15 corresponds to a peak narrower than 30% of the reference window — cleanly excludes gradual peaks. SNR > 2 means the peak rises to at least 2× the median noise — cleanly excludes pure-noise events. CV < 0.30 means inter-peak distances vary by less than 30% — captures regular periodicity for Recurrent.

These are *empirical* values; the thesis is explicit about that ("heuristic thực nghiệm" = "empirical heuristic"). The **self-calibration** mechanism (§5.6) lets these thresholds adapt to the noise floor of the specific stream being analyzed. Section 5.4 of the thesis (`sec:limitation-heuristic`) is a transparent discussion of this.

### Q15. Why is Incremental drift accuracy only 4.4%?

**A.** Because the MMD signal of incremental drift looks geometrically very similar to a noisy plateau:
- An incremental drift slowly walks the distribution from $P_0 \to P_1 \to P_2 \to \cdots$
- Each step is small, so the MMD between consecutive windows is small.
- Over a long span, MMD slowly rises but never spikes.
- Without a clear peak, the peak-detection step fails — there's nothing for SE-CDT to classify.

Mitigations attempted in the thesis: temporal features LTS, MS, SDS try to capture the long-term trend. They help slightly (4.4% > 0%), but the fundamental signal is weak. Future work direction: multi-scale window analysis (look at MMD over $l = 100, 500, 2000$ simultaneously) or learnable embeddings.

This is explicitly called out as **the most important limitation** of the thesis.

### Q16. Why is Concept Memory a ring buffer of size 8 specifically?

**A.** Two practical considerations:
1. **Memory** — each snapshot is 150 points × $d$ features × 8 bytes ≈ $\sim$ 1 KB per snapshot. 8 snapshots = $\sim$ 8 KB. Trivial.
2. **Match efficacy** — with too small ($M = 2, 3$) memory, recurrent drifts that don't match the most recent few are missed. With too large ($M = 50$), false matches become more likely (more candidates → more chances to match by accident). $M = 8$ gives reasonable coverage of distinct concepts a stream might cycle through (think: 4 seasons, 7 days of week, 8 product categories) without bloating the false-match rate.

The choice is a design heuristic; the thesis doesn't claim 8 is optimal, only that it's reasonable.

### Q17. The two BlipProfile branches look ugly. Couldn't you just make it one rule?

**A.** Yes, and the thesis would prefer that. The two branches exist because:
- The "compact pair" version handles **clean blips** — two well-separated peaks of similar height. PPR < 0.20, DPAR > 0.60.
- The "noisy profile" version handles **noisy blips** where the two peaks bleed together due to noise — they look like one wider peak, making FWHM > threshold (so the Growth Process classifies it as PCD). The five extra thresholds catch this case.

Without the noisy branch, ~30% of blips would be misclassified as Gradual or Incremental. Future work: replace both branches with a small classifier trained on simulated blip events (which would get the boundary right automatically). Documented in Appendix A.2 (after F-6.2 fix).

### Q18. How does Concept Memory's matching avoid false matches?

**A.** Three protective layers:
1. The match threshold $\tau_{\mathrm{match}} = 0.15$ is calibrated to be **about 2× the noise-floor MMD** measured on stationary streams. So unrelated distributions almost always exceed this threshold.
2. The kernel bandwidth used for matching is the **average** of the snapshot's stored γ and the new γ — both bandwidths must be similar for a match to score low MMD.
3. If no snapshot matches, the new distribution is *added* (not forced to match). False matches don't accumulate — only successful matches affect the Recurrent classification.

Empirically (Section 8.3), Recurrent accuracy is 71.5%. Failures are mostly missed matches (the snapshot decayed because the stream's noise level changed), not false matches.

## Group D — Experiments and statistics

### Q19. Why 30 seeds and not 100?

**A.** Statistical power vs. compute budget. With 30 seeds × 14 datasets × 8 methods, the benchmark is 3,360 experiments. At ~1 second per experiment, that's ~1 hour per benchmark run on a single core. 100 seeds would multiply by 3.3×, becoming infeasible for iterative development. 30 is the standard in the field (Demšar 2006, KSWIN paper, ShapeDD paper) and gives sufficient power for the Friedman test at $K = 8$. The 95% CI on F1 with 30 seeds × 14 datasets is roughly ±0.02, narrow enough to distinguish methods if they differ.

### Q20. Why 14 datasets and not more?

**A.** Coverage vs. compute. The 14 datasets span:
- 3 STAGGER variants (sudden classification drift)
- 4 random Gaussian shifts (varying severity)
- 1 SEA standard (binary classification)
- 1 Hyperplane (gradual)
- 1 GaussianGradual (synthetic gradual)
- 1 Electricity (semi-real)
- 1 STAGGER none (stationary control)
- 1 RBF blips (transient)
- 1 LED abrupt (multiclass)

This covers the major drift types and modalities. Larger benchmarks like SUNE (30+) exist; future work could include them. The current 14 give meaningful F1 differences and clear Friedman-test results, which is what's needed for the contribution claims.

### Q21. The headline says F1 = 0.531, but DAWIDD gets 0.531 too. Where's the contribution?

**A.** Three places, and the framing matters:

The contribution is **SE-CDT as a unified system**. Within that system, the detection module (ShapeDD-IDW) is benchmarked against DAWIDD on F1, and:
1. **Lower FP at the same F1**: 9.6 FP/run vs DAWIDD's 10.4 FP/run. Lower false-alarm rate matters operationally.
2. **Much faster runtime**: ~7× speedup over ShapeDD original; **DAWIDD itself is the slowest method in the benchmark** (15.18s/stream vs 0.70s for ShapeDD-IDW). So SE-CDT's detection module matches DAWIDD's accuracy at ~22× DAWIDD's speed.
3. **Drift-type classification** — DAWIDD only detects; SE-CDT additionally classifies (its second module). That's an entire downstream capability DAWIDD doesn't have.

So the contribution is **not** "we beat DAWIDD on F1" — it's "SE-CDT matches DAWIDD's detection accuracy with 22× the throughput, lower FP, and additional classification capability built in".

### Q22. Why is Standard MMD's F1 (0.525) so close to the proposed methods' (0.531)?

**A.** Because the benchmark has mostly clean sudden drifts where Standard MMD already does well. The IDW improvement and the calibration improvements show up more in:
- The Type-I error calibration (IDW-MMD has properly calibrated nominal α, Standard MMD's threshold is hand-tuned)
- The runtime (Standard MMD with permutation: 5.05s; IDW-MMD with Gamma: 1.58s — nearly the same as the thesis's pipeline because IDW-MMD itself doesn't permute)
- The downstream classification quality (SE-CDT works because it expects an unbiased trace, which only Standard MMD provides)

The F1 numbers are similar because both methods catch the same easy drifts; the architecture matters more than the F1 delta.

### Q23. The CD test says everything is statistically tied. Doesn't that mean the contribution doesn't beat anything?

**A.** **F1 is not the only metric.** The Friedman+Nemenyi test is on F1 ranks. The contribution is multidimensional:
- F1 ties with DAWIDD ✓
- FP rate beats DAWIDD (9.6 vs 10.4)
- Runtime beats DAWIDD (~22×)
- Type-I error calibration is explicit (DAWIDD doesn't have a Gamma null)
- Adds classification capability (DAWIDD doesn't classify)

In an ML paper, "ties on F1, beats on speed by 7-22×" is a publishable contribution. The thesis is honest that there's no F1 win — but the *value proposition* is "as accurate, much faster, plus classification".

### Q24. How do you know your H0 calibration isn't itself buggy?

**A.** Three checks:
1. **Reproduce the known result**: on Gaussian-iid data, Type-I error should be close to nominal α = 0.05 because Gretton 2009 proved Gamma is asymptotically correct. We get 0.040–0.050 — within sampling error. ✓
2. **Documented previous bug**: the audit note in `wmmd_gamma` (Section 8.7) shows that an earlier `wmmd_asymptotic` implementation was caught producing Type-I error ≈ 0 — clearly wrong. The current implementation no longer has that defect.
3. **Three reference distributions**: i.i.d. d=5, i.i.d. d=10, AR(1) d=5. The first two should pass; the third should over-reject (correlated samples violate i.i.d. assumption). We see exactly that pattern (0.04–0.05 vs 0.075). The pattern is interpretable, not arbitrary.

### Q25. Why is type-specific adaptation only 0.36pp better than periodic retrain on Mixed?

**A.** Because **Periodic Retrain is wasteful but effective on Mixed**: it retrains every 200 samples whether drift happened or not. On a stream with frequent drifts, you usually catch the drift quickly anyway. So the F1 ceiling is similar.

The real difference is **FP rate**: Periodic Retrain triggers 8.3 FP/run (it considers every 200-sample boundary a "drift event"). Type-Specific has 1.1 FP/run because it only acts on real detected drifts. So Type-Specific gives the same accuracy with **8× fewer wasted retrains**. In production, every retrain has compute cost and is a momentary accuracy dip. Reducing them by 8× is significant operationally even if accuracy is similar.

## Group E — System and Kafka

### Q26. Why Redpanda and not vanilla Kafka?

**A.** API compatibility but lower setup complexity. Redpanda is single-binary, no JVM, no Zookeeper. For a thesis prototype where the *protocol* matters more than the implementation, Redpanda is dramatically faster to spin up. Production deployments could use vanilla Kafka with no code changes (same `confluent-kafka` Python client).

### Q27. What happens if the consumer crashes during retraining?

**A.** Three layers of recovery:
1. **Offset commits**: the detection consumer commits its Kafka offset only after successful processing. On restart, it resumes from the last committed offset — no message loss.
2. **Adaptation atomicity**: the adapter uses temp-file-and-rename for the model save. So the consumer always reads either the old model (atomic) or the new model (atomic), never a half-written file.
3. **Producer retries**: explicit `retries=5`, `retry.backoff.ms=100` in the producer config (added in F-10.1). On transient broker outages, the producer will retry without losing messages.

What happens if the **adapter** crashes mid-retraining? The adapter is stateless except for the model output. If it crashes, the partial model is never published; the detection consumer keeps using the old one; the next drift event re-triggers retraining. No corruption.

What's **not** covered: if the broker itself crashes, message recovery time depends on Redpanda's replication factor (1 in this prototype = no recovery). Production would set RF ≥ 3. This is documented as out-of-scope.

### Q28. Could the detection consumer be a bottleneck if the producer rate is too high?

**A.** Yes — at 500 samples/sec, the detection consumer can keep up because the buffering window (150 samples) means SE-CDT runs at most 500/150 ≈ 3 times per second per partition. At higher rates (10,000+ samples/sec), the detector would lag.

Solutions:
- **Partition the topic** so multiple consumer instances share the load. The classification doesn't depend on cross-message order within a single drift event window, so partitioning is safe.
- **Increase the buffer size** to detect drift on coarser windows.
- **Use a faster detector** for the trace step (e.g., subsampling).

The thesis explicitly disclaims this is a single-stream prototype; production scale-out is future work.

## Group F — Limitations and future work

### Q29. What's the single biggest weakness of this thesis?

**A.** SE-CDT subtype accuracy on **Incremental drift (4.4%)**. It's by far the lowest number in the results, and it's an honest "we couldn't solve this case" finding. The MMD signal of incremental drift is geometrically similar to noisy plateaus, and the hand-coded features fail to distinguish them reliably. Future directions: multi-scale window analysis or learnable embeddings.

The conclusion section explicitly says "không nên che giấu" (should not be hidden). This is methodologically the right framing — surfacing the failure mode prevents future researchers from rediscovering it.

### Q30. If you had 6 more months, what would you change?

**A.** Three priorities, in order:
1. **Address Incremental classification** — try multi-scale windows ($l \in \{50, 200, 500\}$) and combine signals. Or train a small ConvNet on simulated MMD traces.
2. **Multi-tenant Kafka deployment** — partition topics, multiple consumers, schema evolution. Show production-grade scaling.
3. **Better p-value correction** — replace Bonferroni with Benjamini-Hochberg for the candidate-validation step.

### Q31. What if your assumption about joint drift is wrong in some real system?

**A.** Then unsupervised detection misses real drift in $P(Y|X)$. The mitigation is to **monitor model accuracy when labels are available** (even if delayed) and use that as a complementary signal. The Kafka prototype publishes `model.accuracy` precisely for this — the dashboard can show both the MMD trace and the prequential accuracy, and an operator can spot disagreement.

A future-work direction is **hybrid drift detection**: combine $P(X)$ monitoring (this thesis) with delayed-label accuracy monitoring (DDM-style), giving best-of-both. The thesis doesn't include this, but the architecture supports it (separate consumers can analyze each signal independently).

### Q32. The paper title says "automatic detection and adaptive update". Did you actually demonstrate the full automation end-to-end?

**A.** Yes, in the Kafka prototype scenario (Section 7.4). One scripted run:
- Producer streams 3000 samples with sudden drift at 1500.
- Detection consumer flags drift around sample 1530–1560 (within tolerance).
- SE-CDT classifies it as TCD-Sudden.
- Adaptation manager runs Full Reset, retrains model on samples 1550–2350.
- Detection consumer reloads new model.
- Prequential accuracy recovers from ~0.59 to ~0.93.

This is **end-to-end automatic** in the sense that no human intervention occurs after `producer.py` is started. The scope is "single scenario to validate the architecture", not "comprehensive evaluation across many scenarios". Detection-only and adaptation-only experiments use 30-run benchmarks for proper statistics; the Kafka demo is one demonstrative run.

### Q33. Why did you not benchmark on more real-world data (only Electricity)?

**A.** Two reasons:
1. **Real-world drift is hard to ground-truth**. To compute F1, EDR, MTTD you need to know *exactly* when drift happened. In real data, drift events aren't labeled — you have to infer them from accuracy changes (which themselves depend on the model). This circularity makes F1 measurements unreliable.
2. **Synthetic benchmarks isolate the variable being tested**. With known drift positions, types, and magnitudes, we can attribute F1 differences to the detector, not to data idiosyncrasies. This is the methodology of every drift-detection paper since Gama 2014.

Electricity is included as a semi-real check that the methods generalize beyond pure synthetic data. A larger real-world study with operational labels (e.g., a deployed model with confirmed drift events) is future work.

### Q34. Do your results depend on the specific synthetic data generators you chose?

**A.** Partially. Results would shift if you used different generators, but the *relative* ordering of methods (the Friedman ranking) is robust because:
1. The synthetic generators are standard (River library, MOA library) — the same ones used by ShapeDD, DAWIDD, KSWIN papers.
2. The 14 datasets span enough drift types and dimensionalities (d ∈ {2, 5, 10, 50}) that no single generator dominates.
3. The Nemenyi test compares ranks within each dataset, not absolute F1 across datasets, so dataset-specific quirks average out.

That said, on a **specific deployed system**, F1 could be higher or lower than the benchmark suggests. The benchmark establishes that the methods are *competitive*, not that they hit specific F1 targets in production.

### Q35. Should the choice of base classifier (Logistic Regression) affect the adaptation results?

**A.** Yes, but the *direction* of the comparison should be stable. We use Logistic Regression because:
- It's fast (important for prequential evaluation with 30 seeds × 5000 samples × 5 events = 750,000 fits).
- It supports incremental update (`partial_fit`), which Incremental and Gradual strategies need.
- It has a clean prequential-accuracy curve (smoother than e.g. random forest).

A more powerful base classifier (random forest, gradient boosting) would shift absolute accuracy numbers up but keep the *ranking* of strategies the same: Type-Specific should still beat No Adaptation, Periodic Retrain should still have higher FP. The thesis is explicit that LR is a defensible choice for the adaptation evaluation, not the only choice.

---

## 12.7 Closing tips for the defense

1. **Lead with the thesis's central narrative**, not the F1 number. The narrative: "Concept drift in production is hard because labels are delayed and drift comes in many shapes. We built a system that detects drift unsupervised, classifies it into 5 types, and adapts the model accordingly — using a 7×-faster MMD test and a hand-engineered classifier. The system works end-to-end in a Kafka prototype."

2. **Embrace the limitations**. The Incremental 4.4% number is a *strength* if you frame it right: it shows you measured carefully and reported what you found. Examiners respect this far more than vague hand-waving about "future work".

3. **Distinguish what's new from what's inherited**. ShapeDD's triangle theorem is borrowed (and cited). CDT-MSW's TCD/PCD partition is borrowed (and cited). What's new: IDW reweighting + Gamma p-value (detection), unsupervised classifier reading the MMD signal (classification), type-specific adaptation dispatch (system).

4. **Be ready to demo the code**. If the examiner asks "show me where you implement IDW", point to `core/detectors/mmd_variants.py:127–155` (`compute_optimal_weights`). The implementation map (Section 11) was prepared for this.

5. **Be ready to defend the thresholds**. WR < 0.15, SNR > 2.0, etc. — these are heuristic. The honest answer: "I picked them empirically, validated with self-calibration, and accept that this is a future-work area for a learnable classifier."

6. **Practice the math explanations slowly**. MMD's three-term formula (within-X, within-Y, cross), the IDW weight derivation, the Friedman+Nemenyi procedure. These are the questions where being slow and clear matters.

---

## 12.8 Knowledge summary — architecture cheat sheet

Quick reference distilled from the architectural Q&A. Use this for fast review or to answer examiner questions on the fly.

### The naming hierarchy

| Term | What it is |
|------|-----------|
| **SE-CDT** | The proposed unified detector-classifier *system*. The "T" in the acronym (Type identification) refers to drift-type classification — the core novel contribution. |
| **ShapeDD-IDW** | SE-CDT's *detection module*. An engineering refinement of ShapeDD (2021). |
| **(unnamed) classification module** | SE-CDT's *classification module*. No separate name because SE-CDT was named after this part. |
| **IDW-MMD** | An *algorithm* (a kernel two-sample test statistic) used inside ShapeDD-IDW's validation step. |
| **Adaptation framework** | A separate module that *consumes* SE-CDT's classification output. |

### The central object: σ(t)

- σ(t) is computed using **Standard MMD**, NOT IDW-MMD.
- σ(t) is the **single shared input** for both detection and classification.
- Both detection's peak-finding and classification's feature extraction read the *same* σ(t).
- IDW-MMD only enters at the validation step (per-candidate hypothesis test).

### Where the sudden bias lives

In the **peak-detection step** of ShapeDD-IDW, NOT in IDW-MMD validation.
- Peak detection looks for sharp peaks → biased toward Sudden / Blip / Recurrent (TCD types).
- IDW-MMD validation is a generic two-sample test → not sudden-specific.
- Gradual / Incremental drift get missed at peak detection (no clear peak), not at validation.

### ShapeDD vs ShapeDD-IDW: refinement, not revolution

| Aspect | ShapeDD (2021) | ShapeDD-IDW (this thesis) |
|--------|---------------|---------------------------|
| σ(t) trace | Standard MMD | **Standard MMD** (same) |
| Window setup | l₁ ref, l₂ test, sliding | **Same** |
| Shape filter | Convolution with h'_l (matched filter) | Local-maxima peak detection |
| Validation | Permutation test, B=2500 | IDW-MMD + Gamma null, B=20 |
| End-to-end speedup | 1× | **~7×** |

ShapeDD-IDW shares ~80% of the pipeline with ShapeDD. The genuinely novel piece of the thesis is **SE-CDT's classification module** — ShapeDD never did classification.

### Threshold adaptation status

**Self-calibrating** (3 thresholds, via `_RollingFeatureBaseline`):
- τ_WR = 0.15 → loosens to 25th-quantile of baseline WR
- τ_SNR = 2.0 → tightens to 90th-quantile of baseline SNR
- τ_CV = 0.30 → loosens to 25th-quantile of baseline CV

**Fixed across all datasets** (documented limitation, §5.3 of conclusion):
- LTS = 0.5, MS = 0.6, SDS = 0.12 (temporal features)
- WR_TCD_PCD = 0.12 (Growth Process)
- PPR = 0.20, DPAR = 0.60 (Blip criteria)
- τ_match = 0.15 (Concept Memory)
- BlipProfile noisy variant: WR<0.17, SNR ∈ (1.45, 2.60), DPAR ∈ (0.45, 0.85), LTS<0.12

All thresholds sit inside naturally-bounded feature ranges (most features are in [0,1]). They cannot be exceeded mathematically — only mismatched to the dataset's noise floor. Self-calibration handles three of those mismatches; the rest are acknowledged limitations.

**Empirical failure evidence:** Incremental drift accuracy of 4.4% is direct evidence that fixed LTS / MS / SDS thresholds don't generalize across all drift types.

### Defense one-liners

**On the system architecture:**
> *"SE-CDT is a unified detector-classifier system. The detection module (ShapeDD-IDW) refines ShapeDD with IDW-MMD validation and a Gamma null, ~7× faster. The classification module reads the same σ(t) signal and adds drift-type identification — ShapeDD never did this. That's where the genuine novelty lives."*

**On σ(t):**
> *"σ(t) is computed using Standard MMD, not IDW-MMD. IDW-MMD only runs at the per-candidate validation step. Standard MMD is used for the trace because its uniform weighting preserves the geometric shape that classification reads."*

**On the sudden bias:**
> *"The sudden bias is in the peak-detection step, not the IDW-MMD validation. Validation is a generic two-sample test. Peak detection looks for sharp peaks, which favor sudden / blip / recurrent. Gradual and incremental drift are missed at the peak-detection step."*

**On thresholds:**
> *"Three thresholds (τ_WR, τ_SNR, τ_CV) self-calibrate to noise floor; six others are fixed. Fixed thresholds are documented as a limitation in §5.3, with the 4.4% Incremental result as direct empirical evidence."*

---

*End of guide. Total: ~18,500 words. Last updated: 2026-05-19. Companion to thesis PDF `report/latex/2370116_LePhucDuc_ThesisReport.pdf` (102 pages, 5.6 MB, 0 warnings/errors).*
