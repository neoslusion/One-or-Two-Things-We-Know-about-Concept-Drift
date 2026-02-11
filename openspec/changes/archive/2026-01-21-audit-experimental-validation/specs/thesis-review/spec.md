## ADDED Requirements

### Requirement: Experimental Validity
The thesis experimental section MUST be validated against original literature and standard benchmarks.

#### Scenario: Literature Consistency
- **WHEN** a baseline method (e.g., ADWIN) is described
- **THEN** the description and parameter settings MUST match the original paper
- **AND** any deviations MUST be justified.

#### Scenario: Reproducibility Check
- **WHEN** the experimental setup is audited
- **THEN** the data pipeline (preprocessing, splitting) MUST be clearly documented
- **AND** random seeds MUST be fixed for reproducibility.

### Requirement: Artifact Integrity
All tables and figures MUST be accurate, informative, and consistent with the text.

#### Scenario: Table Correctness
- **WHEN** a results table is presented
- **THEN** the metrics (F1, Accuracy) MUST be correctly calculated
- **AND** the best results MUST be correctly highlighted
- **AND** the table content MUST match the analysis in the text.
