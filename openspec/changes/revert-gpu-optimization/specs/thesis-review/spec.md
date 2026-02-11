## MODIFIED Requirements

### Requirement: Hardware Consistency
All drift detection methods MUST execute on the same hardware (CPU) to ensure fair comparison and optimal performance for small window sizes.

#### Scenario: Small Window Benchmark
- **WHEN** running the benchmark with `CHUNK_SIZE=150`
- **THEN** all methods (`MMD`, `ShapeDD`, `KS`, `D3`) MUST utilize CPU execution
- **AND** execution time for `ShapeDD` MUST be comparable to or faster than GPU execution for this window size.
