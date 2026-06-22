# 4B sparsity-sweep — relative edge % (RBF − hybrid)/RBF, overall test MAE

Positive = hybrid better. Rung 0 = dense (full context).

| regime | r0 (i=0.0) | r1 (i=0.2) | r2 (i=0.4) | r3 (i=0.6) | r4 (i=0.8) |
|---|---|---|---|---|---|
| fewer_quotes | +2.05% | +2.61% | +3.18% | +3.80% | +4.32% |
| thin_wings | +2.05% | +0.77% | +0.16% | +0.11% | +0.31% |
| missing_maturities | +2.05% | +2.73% | +3.10% | +2.49% | +1.56% |
| combined_quotes_wings | +2.05% | +0.96% | +0.30% | +0.26% | +0.46% |

**Gate verdict:** `accuracy_survives = ambiguous`

a trend is present (relative edge exceeds the 2% dense band in at least one regime) but the wing-sensitive regimes do not clear the >=5% survive bar; the eval-time fairness caveat (full-context checkpoint scored on thinner, partly OOD context) clouds the wing read -> trigger the conditional fair-retrain escalation 4B.7.
