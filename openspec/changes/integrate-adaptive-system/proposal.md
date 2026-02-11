# System Integration for Adaptive Learning Thesis

## Summary
This proposal integrates existing disjoint components (Drift Detection, Classification, Adaptation) into a unified "Adaptive Learning System" framework suitable for a Master's thesis. It formalizes the architecture, standardizes terminology, and identifies gaps for a complete academic publication.

## Problem Statement
The current codebase contains isolated components (`drift_detection_benchmark`, `drift_monitoring_system`) with inconsistent naming and loose coupling. There is no clear "System" definition linking Detection -> Classification -> Adaptation logically for the thesis.

## Goals
1.  **Formalize Architecture:** Define the 5-module adaptive pipeline (Data, Model, Detector, Analyzer, Adaptor).
2.  **Standardize Terminology:** Unify naming (e.g., `shape_with_wmmd` -> `shape_ow_mmd`, "Drift Type Classification").
3.  **Validate Logic:** Ensure the `evaluate_prequential.py` script reflects the "System" described in the thesis.
4.  **Gap Analysis:** Identify missing experiments (Sensitivity, Overhead) for a top-tier paper.

## Non-Goals
-   Developing new drift detection algorithms.
-   Building a production-grade distributed system (focus is on the logical research system).
