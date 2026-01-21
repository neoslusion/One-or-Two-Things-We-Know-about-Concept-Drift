# Experimental Validation Audit Design

## 1. Literature Cross-Reference Strategy
The audit will involve direct verification against primary sources.
- **Tools:** Use `webfetch` (if available/simulated via tool) or manual search knowledge to verify papers. Since I cannot browse the live web, I will use my internal knowledge base of concept drift literature (Gama, Bifet, etc.) and `grep` the local bibliography (`report/latex/references.bib`) to identify keys.
- **Process:**
    1.  Extract citation keys from `04_experiments_evaluation.tex`.
    2.  Check `references.bib` for metadata.
    3.  Verify claims against standard knowledge of these papers (e.g., DDM uses binomial distribution, ADWIN uses Hoeffding bound).

## 2. Deep Experimental Audit
- **Data Pipeline:** Inspect `experiments/drift_detection_benchmark/datasets/` and `experiments/benchmark_proper.py` to trace data flow.
- **Protocol:** Check `run_benchmark.sh` and `experiments/drift_detection_benchmark/main.py` for random seeds, loops, and metric calculations.

## 3. Output Artifacts
- **Audit Report:** `report/experimental_validation_audit.md` containing the specific sections requested (Literature, Setup, Tables/Figures, Gaps).
- **Issue Tracker:** Add critical issues to `tasks.md` for immediate resolution if they are simple (e.g., typo in citation). Complex issues will be documented in the report.
