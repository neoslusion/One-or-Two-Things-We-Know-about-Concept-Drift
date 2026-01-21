# thesis-review Specification

## Purpose
TBD - created by archiving change review-thesis-comprehensive. Update Purpose after archive.
## Requirements
### Requirement: Thesis Quality Standards
The thesis report SHALL meet the following quality standards before defense:
1. All technical terminology MUST be consistent throughout the document
2. All numerical data in tables and text MUST be reconciled
3. All forward references MUST include brief context
4. All encoding errors MUST be fixed

#### Scenario: Terminology consistency check
- **WHEN** a method name (e.g., SHAPED_CDT) is introduced
- **THEN** all subsequent references to that method use the same name
- **AND** abbreviations are explicitly defined on first use

#### Scenario: Data consistency check
- **WHEN** the same metric (e.g., CAT Accuracy) is reported in multiple places
- **THEN** each occurrence includes context explaining the measurement conditions
- **AND** any discrepancies are explicitly explained

### Requirement: Kafka POC Demonstration
The thesis SHALL either:
1. Include visual evidence of Kafka streaming system operation (screenshots, metrics), OR
2. Explicitly limit the scope to "architecture design" without implementation claims

#### Scenario: Full POC demonstration
- **WHEN** the thesis claims implementation of Kafka streaming system
- **THEN** Chapter 4 includes at least one screenshot of the running system
- **AND** basic metrics (throughput, latency) are reported

#### Scenario: Design-only scope
- **WHEN** the thesis limits scope to architecture design
- **THEN** Chapter 0 objectives are updated to reflect this
- **AND** Chapter 3 clearly states "design proposal, not implemented"

### Requirement: Technical Accuracy
The thesis methodology and implementation MUST be technically accurate and consistent with source papers.

#### Scenario: ADW-MMD formula verification
- **WHEN** ADW-MMD formula is presented in Chapter 3
- **THEN** the formula matches the implementation in code
- **AND** any adaptations from original paper are explicitly noted
- **AND** the cross-term weighting (uniform vs weighted) is clearly specified

#### Scenario: Citation accuracy
- **WHEN** a method is attributed to a paper
- **THEN** the implementation follows that paper's specification
- **OR** differences are explicitly documented as "adapted" or "inspired by"

### Requirement: Comprehensive Review Report
A detailed audit report SHALL be generated in `report/thesis_review_audit.md` covering all aspects of the thesis.

#### Scenario: Report Structure
- **WHEN** the review is complete
- **THEN** the report contains sections for Research Foundation, Methodology, Data & Experiments, Results & Analysis, Writing, and Technical Correctness
- **AND** each section includes Status, Findings, Suggestions, and Priority.

### Requirement: Research Foundation Quality
The review MUST assess the clarity of the problem statement and the comprehensiveness of the literature review.

#### Scenario: Problem Statement Check
- **WHEN** reviewing Chapter 1
- **THEN** verify if the problem (concept drift in unsupervised streams) is clearly defined
- **AND** verify if the motivation (why current methods fail) is compelling.

### Requirement: Methodology Soundness
The review MUST assess the logical soundness and novelty of the proposed method (SE-CDT and ADW-MMD).

#### Scenario: Method Logical Flow
- **WHEN** reviewing Chapter 3
- **THEN** verify if the transition from ShapeDD to SE-CDT is logical
- **AND** verify if the ADW-MMD adaptation is mathematically justified.

### Requirement: Experimental Rigor
The review MUST assess the appropriateness of the data, experimental setup, and evaluation metrics.

#### Scenario: Baseline Fairness
- **WHEN** reviewing Chapter 4
- **THEN** verify if baselines (CDT_MSW, MMD, etc.) are compared fairly
- **AND** verify if metrics (F1, CAT Accuracy) are appropriate for the task.

### Requirement: Technical Correctness
The review MUST verify that the equations in the thesis match the code implementation.

#### Scenario: Equation-Code Consistency
- **WHEN** checking equations in Chapter 3
- **THEN** compare them with `experiments/backup/mmd_variants.py`
- **AND** report any discrepancies.

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

