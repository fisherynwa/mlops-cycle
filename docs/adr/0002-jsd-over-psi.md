# ADR 0002 — Jensen-Shannon divergence for drift

## Context
We need a numeric drift metric with an interpretable, stable threshold that works
for both features and model outputs.

## Decision
Use Jensen-Shannon distance (base 2, bounded [0,1], threshold 0.10) as the primary
numeric drift statistic, over PSI, KS, or raw KL.

## Consequences
- (+) Symmetric and bounded, so a single fixed threshold is meaningful and
  comparable across columns.
- (+) Finite even when a histogram bin is empty in one distribution (raw KL is not).
- (+) Directly supported by Evidently (`num_method="jensenshannon"`).
- (−) Requires binning continuous features; results depend on bin count, so bins
  are held fixed between reference and current.
- Note: PSI (the insurance/credit-risk standard, banded 0.1/0.25) is reported as a
  familiar cross-check but is not the gate.
