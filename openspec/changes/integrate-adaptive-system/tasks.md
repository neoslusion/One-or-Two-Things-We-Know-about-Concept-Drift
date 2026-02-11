# Tasks

- [x] Refactor `experiments/backup/se_cdt.py` to encapsulate `shapedd_adw_mmd` (from `mmd_variants.py`) inside `SE_CDT` class. <!-- id: 0 -->
- [x] Implement `monitor(window)` method in `SE_CDT` for continuous detection. <!-- id: 1 -->
- [x] Refactor `evaluate_prequential.py` to use the unified `se_cdt.monitor(window)` interface (removing manual slicing). <!-- id: 2 -->
- [x] Add `classification_time` timing to `evaluate_prequential.py` main loop. <!-- id: 3 -->
- [x] Add CLI arguments (`--w_ref`, `--sudden_thresh`) to `evaluate_prequential.py`. <!-- id: 4 -->
- [x] Add module-level docstrings to `evaluate_prequential.py` explicitly mapping to Design modules. <!-- id: 5 -->
- [x] Create a `run_sensitivity_analysis.sh` script to sweep parameters. <!-- id: 6 -->
