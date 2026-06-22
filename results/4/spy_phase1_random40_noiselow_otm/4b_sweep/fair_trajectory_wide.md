# 4B.7 fair retrain — relative edge % (RBF − hybrid)/RBF vs the 4B.5 eval-time read

Positive = hybrid better. Rung 0 = dense (4A.4 anchor, not retrained).

| regime | rung | eval-time edge | **fair edge** | fair sig |
|---|---|---|---|---|
| combined_quotes_wings | r0 (i=0.0) | +2.05% | **+2.05%** | True |
| combined_quotes_wings | r1 (i=0.2) | +0.96% | **+2.38%** | True |
| combined_quotes_wings | r2 (i=0.4) | +0.30% | **+5.39%** | True |
| combined_quotes_wings | r3 (i=0.6) | +0.26% | **+14.08%** | True |
| combined_quotes_wings | r4 (i=0.8) | +0.46% | **+15.99%** | True |
| fewer_quotes | r0 (i=0.0) | +2.05% | **+2.05%** | True |
| fewer_quotes | r1 (i=0.2) | +2.61% | **+2.79%** | True |
| fewer_quotes | r2 (i=0.4) | +3.18% | **+1.94%** | True |
| fewer_quotes | r3 (i=0.6) | +3.80% | **+0.14%** | True |
| fewer_quotes | r4 (i=0.8) | +4.32% | **+0.07%** | False |
| missing_maturities | r0 (i=0.0) | +2.05% | **+2.05%** | True |
| missing_maturities | r1 (i=0.2) | +2.73% | **+1.37%** | True |
| missing_maturities | r2 (i=0.4) | +3.10% | **+9.53%** | True |
| missing_maturities | r3 (i=0.6) | +2.49% | **+20.91%** | True |
| missing_maturities | r4 (i=0.8) | +1.56% | **+37.07%** | True |
| thin_wings | r0 (i=0.0) | +2.05% | **+2.05%** | True |
| thin_wings | r1 (i=0.2) | +0.77% | **+1.75%** | True |
| thin_wings | r2 (i=0.4) | +0.16% | **+5.25%** | True |
| thin_wings | r3 (i=0.6) | +0.11% | **+9.98%** | True |
| thin_wings | r4 (i=0.8) | +0.31% | **+16.57%** | True |

**Resolved gate verdict (was `ambiguous` eval-time):** `accuracy_survives = true`

wing-sensitive regimes show a growing, significant, >=5% relative edge at the sparse end — accuracy survives sparsity.
