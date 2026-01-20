# Unified Detector-Classifier Specification

## 1. SE-CDT System Refactoring
### ADDED Requirements
#### Scenario: Unified Detection & Classification
- **Code:** Update `SE_CDT` class in `experiments/backup/se_cdt.py`.
- **Logic:** Add a `monitor(window)` method that:
    1.  Calls `shapedd_adw_mmd(window)` to get `pattern_score` and `mmd_trace`.
    2.  If `pattern_score > threshold`: Calls `self.classify(mmd_trace)`.
    3.  Returns a unified result object: `{is_drift: bool, type: str, score: float, ...}`.
- **Reason:** Encapsulates the complexity, matching the `CDT_MSW` design pattern.

## 2. Prequential Evaluation Update
### MODIFIED Requirements
#### Scenario: Use Unified System
- **Code:** Update `evaluate_prequential.py` loop.
- **Logic:** Instead of manually computing MMD $\rightarrow$ Slicing $\rightarrow$ Classifying, simply call `se_cdt_instance.monitor(current_window)`.
- **Benefit:** Cleaner code, less prone to "manual slicing" errors.

## 3. Metrics & Benchmarking
### ADDED Requirements
#### Scenario: Classification Time Tracking
- **Code:** Measure `time.time()` inside `SE_CDT.monitor()` specifically for the classification step (only when drift is detected).
- **Reason:** Prove that the "Heuristic Classification" is fast enough for real-time streams.
