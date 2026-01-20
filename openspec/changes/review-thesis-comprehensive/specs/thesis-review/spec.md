## ADDED Requirements

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

## MODIFIED Requirements

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
