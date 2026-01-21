## MODIFIED Requirements

### Requirement: Table Generation Consistency
Automated analysis scripts MUST generate LaTeX tables that strictly match the format, metric selection, and style of the thesis tables.

#### Scenario: Aggregate Table Generation
- **WHEN** the `latex_export.py` script is executed
- **THEN** it MUST output a `table_comparison_aggregate.tex` file
- **AND** the columns MUST include: Precision, Recall (EDR), F1, Delay, FP
- **AND** the values MUST be formatted to 3 decimal places (e.g., 0.854)
- **AND** the best results per column MUST be bolded.
