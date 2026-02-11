# workspace-consolidation Specification

## Purpose
Ensure the workspace is organized according to a standard research project structure, with centralized outputs and clear entry points.

## ADDED Requirements
### Requirement: Centralized Result Storage
All experimental outputs MUST be stored in a top-level `results/` directory.

#### Scenario: LaTeX Table Format
- **WHEN** a LaTeX table is exported
- **THEN** it MUST use the `\begin{tabular}{|l|...|}` format with vertical separators
- **AND** it MUST use `\hline` for horizontal lines
- **AND** it MUST NOT require the `booktabs` package
- **AND** it MUST NOT generate a PDF version of the table.

### Requirement: Root Entry Point
The project MUST provide a single entry point at the root for running all major benchmarks.

#### Scenario: Dispatcher execution
- **WHEN** `python main.py benchmark` is run
- **THEN** it executes the comprehensive drift detection benchmark suite.

### Requirement: Algorithm Centralization
All core drift detection algorithms MUST be located in the `core/detectors/` directory.

#### Scenario: Importing a detector
- **WHEN** a script needs to use the ShapeDD algorithm
- **THEN** it imports from `core.detectors.shape_dd`.

### Requirement: Dataset Separation
Data generation and cataloging logic MUST be separated from experiment logic.

#### Scenario: Dataset generation
- **WHEN** a new stream is needed
- **THEN** it is generated using functions in the `data/` package.
