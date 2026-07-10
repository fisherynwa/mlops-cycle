# ADR 0001 — GAM instead of gradient boosting

## Context
The model informs a cost estimate in a domain where the shape of each feature's
effect must be explainable to non-technical stakeholders and, in the real world,
to regulators.

## Decision
Use a Generalized Additive Model (`LinearGAM`) rather than a black-box gradient-
boosted tree ensemble.

## Consequences
- (+) Every feature effect is a plotted, monotonic-checkable curve; we can verify
  the model recovered the true age and BMI relationships.
- (+) Interpretability pairs naturally with drift analysis — a shifted feature has
  a legible impact path.
- (−) Slightly lower raw accuracy than a tuned GBDT would reach; acceptable given
  the interpretability requirement. An EBM (GA2M) is the documented upgrade path
  if pairwise interactions become necessary.
