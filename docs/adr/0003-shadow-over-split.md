# ADR 0003 — Shadow A/B by default, split optional

## Context
A drift-triggered retrain produces a challenger that must be compared to the
champion before promotion. Charges are heavy-tailed, so comparison power matters.

## Decision
Default to shadow scoring (both models score every request; champion is served;
both predictions logged), with split-traffic routing available via `AB_MODE=split`.

## Consequences
- (+) Shadow yields paired errors on identical inputs → a paired test (Wilcoxon)
  with far more power than an independent-sample test on skewed data.
- (+) No user-facing risk, since only the champion is ever served.
- (+) Promotion is a single alias move (`@champion`) gated on a bootstrap CI, a
  significance test, and a minimum-sample guardrail against peeking.
- (−) Doubles inference cost per request; acceptable for this model's size.
- Split mode remains for cases where variants must genuinely serve different users.
