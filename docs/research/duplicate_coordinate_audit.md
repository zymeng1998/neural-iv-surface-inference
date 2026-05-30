# Duplicate-coordinate audit

---
status: executed
type: data_audit
input_strict: data_processed/spy/spy_surface_points_strict.parquet
input_benchmarks: ['spy_phase1_random40_noiselow.parquet']
---

## 1. Verdict

**Severity: SEVERE**.

- 21,072,592 / 22,512,040 rows (93.61%) live inside a duplicate `(date, expiration, strike)` group in the strict surface table.

- 21,072,592 / 22,512,040 rows (93.61%) live inside a duplicate `(date, round(log_m,10), round(tau,10))` group — the key the conditional model and RBF baseline both treat as the surface coordinate.

- Of the contract-level dup groups, 10,530,258 are call-put pairs (100.0%) and 444 are same-type duplicates.

## 2. Exact numbers

### 2.1 Duplicate counts by key definition

| Key | Total rows | Rows in dup groups | % | Dup groups | size=2 | size=3 | size≥4 | Call-put mix | Same-type |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `date+expiration+strike` | 22,512,040 | 21,072,592 | 93.61% | 10,530,702 | 10,525,108 | 0 | 5,594 | 10,530,258 | 444 |
| `date+round(lm,8)+round(tau,8)` | 22,512,040 | 21,072,592 | 93.61% | 10,530,702 | 10,525,108 | 0 | 5,594 | 10,530,258 | 444 |
| `date+round(lm,10)+round(tau,10)` | 22,512,040 | 21,072,592 | 93.61% | 10,530,702 | 10,525,108 | 0 | 5,594 | 10,530,258 | 444 |
| `date+round(lm,12)+round(tau,12)` | 22,512,040 | 21,072,592 | 93.61% | 10,530,702 | 10,525,108 | 0 | 5,594 | 10,530,258 | 444 |

### 2.2 IV-label dispersion within duplicate groups

`iv_range = max(IV) − min(IV)` inside each group.

| Key | Group kind | n | mean | p50 | p90 | p95 | p99 | max |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `date+expiration+strike` | call_put_mix | 10,530,258 | 0.1026 | 0.0488 | 0.3024 | 0.3902 | 0.5951 | 2.9560 |
| `date+expiration+strike` | same_type | 444 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `date+expiration+strike` | all | 10,530,702 | 0.1026 | 0.0488 | 0.3024 | 0.3902 | 0.5951 | 2.9560 |
| `date+round(lm,8)+round(tau,8)` | call_put_mix | 10,530,258 | 0.1026 | 0.0488 | 0.3024 | 0.3902 | 0.5951 | 2.9560 |
| `date+round(lm,8)+round(tau,8)` | same_type | 444 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `date+round(lm,8)+round(tau,8)` | all | 10,530,702 | 0.1026 | 0.0488 | 0.3024 | 0.3902 | 0.5951 | 2.9560 |
| `date+round(lm,10)+round(tau,10)` | call_put_mix | 10,530,258 | 0.1026 | 0.0488 | 0.3024 | 0.3902 | 0.5951 | 2.9560 |
| `date+round(lm,10)+round(tau,10)` | same_type | 444 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `date+round(lm,10)+round(tau,10)` | all | 10,530,702 | 0.1026 | 0.0488 | 0.3024 | 0.3902 | 0.5951 | 2.9560 |
| `date+round(lm,12)+round(tau,12)` | call_put_mix | 10,530,258 | 0.1026 | 0.0488 | 0.3024 | 0.3902 | 0.5951 | 2.9560 |
| `date+round(lm,12)+round(tau,12)` | same_type | 444 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `date+round(lm,12)+round(tau,12)` | all | 10,530,702 | 0.1026 | 0.0488 | 0.3024 | 0.3902 | 0.5951 | 2.9560 |

### 2.3 Benchmark observed-hidden leakage

Held-out rows whose coordinate (rounded to 10 d.p.) coincides with at least one observed row on the same date. The 'twin MAE' column is the MAE of the trivial baseline that predicts the mean observed IV at the exact coordinate; it tells us whether nearest-distance=0 is genuine local density or a duplicate-leg artifact.

| Benchmark | dp | Split | Money | Maturity | Hidden | With twin | % | Twin MAE | iv_range mean |
|---|---:|---|---|---|---:|---:|---:|---:|---:|
| spy_phase1_random40_noiselow | 8 | test | atm | long | 145,965 | 58,069 | 39.78% | 0.0333 | 0.0332 |
| spy_phase1_random40_noiselow | 8 | test | atm | medium | 346,006 | 138,155 | 39.93% | 0.0192 | 0.0175 |
| spy_phase1_random40_noiselow | 8 | test | atm | short | 490,909 | 192,479 | 39.21% | 0.0467 | 0.0451 |
| spy_phase1_random40_noiselow | 8 | test | call_wing | long | 153,487 | 61,212 | 39.88% | 0.0920 | 0.0920 |
| spy_phase1_random40_noiselow | 8 | test | call_wing | medium | 203,692 | 80,821 | 39.68% | 0.1099 | 0.1098 |
| spy_phase1_random40_noiselow | 8 | test | call_wing | short | 145,334 | 34,690 | 23.87% | 0.1443 | 0.1442 |
| spy_phase1_random40_noiselow | 8 | test | deep_call_wing | long | 194,946 | 76,348 | 39.16% | 0.2548 | 0.2548 |
| spy_phase1_random40_noiselow | 8 | test | deep_call_wing | medium | 58,556 | 19,657 | 33.57% | 0.2464 | 0.2464 |
| spy_phase1_random40_noiselow | 8 | test | deep_call_wing | short | 20,305 | 413 | 2.03% | 0.2730 | 0.2733 |
| spy_phase1_random40_noiselow | 8 | test | deep_put_wing | long | 316,824 | 126,175 | 39.82% | 0.2465 | 0.2465 |
| spy_phase1_random40_noiselow | 8 | test | deep_put_wing | medium | 332,945 | 132,689 | 39.85% | 0.2411 | 0.2409 |
| spy_phase1_random40_noiselow | 8 | test | deep_put_wing | short | 154,446 | 46,400 | 30.04% | 0.3572 | 0.3572 |
| spy_phase1_random40_noiselow | 8 | test | put_wing | long | 196,169 | 78,976 | 40.26% | 0.0410 | 0.0408 |
| spy_phase1_random40_noiselow | 8 | test | put_wing | medium | 361,791 | 144,121 | 39.84% | 0.0415 | 0.0403 |
| spy_phase1_random40_noiselow | 8 | test | put_wing | short | 363,759 | 138,881 | 38.18% | 0.1780 | 0.1776 |
| spy_phase1_random40_noiselow | 8 | train | atm | long | 426,888 | 171,038 | 40.07% | 0.0375 | 0.0370 |
| spy_phase1_random40_noiselow | 8 | train | atm | medium | 575,252 | 231,097 | 40.17% | 0.0248 | 0.0234 |
| spy_phase1_random40_noiselow | 8 | train | atm | short | 655,715 | 255,623 | 38.98% | 0.0388 | 0.0368 |
| spy_phase1_random40_noiselow | 8 | train | call_wing | long | 439,488 | 176,255 | 40.10% | 0.0562 | 0.0562 |
| spy_phase1_random40_noiselow | 8 | train | call_wing | medium | 441,926 | 163,874 | 37.08% | 0.0718 | 0.0717 |
| spy_phase1_random40_noiselow | 8 | train | call_wing | short | 238,485 | 60,234 | 25.26% | 0.1169 | 0.1170 |
| spy_phase1_random40_noiselow | 8 | train | deep_call_wing | long | 321,436 | 113,139 | 35.20% | 0.1126 | 0.1127 |
| spy_phase1_random40_noiselow | 8 | train | deep_call_wing | medium | 164,464 | 27,207 | 16.54% | 0.1502 | 0.1505 |
| spy_phase1_random40_noiselow | 8 | train | deep_call_wing | short | 45,809 | 3,429 | 7.49% | 0.4084 | 0.4086 |
| spy_phase1_random40_noiselow | 8 | train | deep_put_wing | long | 736,461 | 293,537 | 39.86% | 0.1875 | 0.1876 |
| spy_phase1_random40_noiselow | 8 | train | deep_put_wing | medium | 791,285 | 291,893 | 36.89% | 0.1859 | 0.1857 |
| spy_phase1_random40_noiselow | 8 | train | deep_put_wing | short | 241,609 | 55,709 | 23.06% | 0.3267 | 0.3272 |
| spy_phase1_random40_noiselow | 8 | train | put_wing | long | 514,528 | 205,749 | 39.99% | 0.0482 | 0.0482 |
| spy_phase1_random40_noiselow | 8 | train | put_wing | medium | 600,304 | 239,938 | 39.97% | 0.0449 | 0.0440 |
| spy_phase1_random40_noiselow | 8 | train | put_wing | short | 476,917 | 176,683 | 37.05% | 0.1158 | 0.1151 |
| spy_phase1_random40_noiselow | 8 | val | atm | long | 155,320 | 61,748 | 39.76% | 0.0231 | 0.0222 |
| spy_phase1_random40_noiselow | 8 | val | atm | medium | 285,366 | 114,486 | 40.12% | 0.0183 | 0.0165 |
| spy_phase1_random40_noiselow | 8 | val | atm | short | 396,851 | 156,916 | 39.54% | 0.0363 | 0.0342 |
| spy_phase1_random40_noiselow | 8 | val | call_wing | long | 160,994 | 64,391 | 40.00% | 0.0386 | 0.0379 |
| spy_phase1_random40_noiselow | 8 | val | call_wing | medium | 243,488 | 95,985 | 39.42% | 0.0626 | 0.0621 |
| spy_phase1_random40_noiselow | 8 | val | call_wing | short | 253,946 | 69,761 | 27.47% | 0.1154 | 0.1151 |
| spy_phase1_random40_noiselow | 8 | val | deep_call_wing | long | 192,651 | 74,710 | 38.78% | 0.1344 | 0.1342 |
| spy_phase1_random40_noiselow | 8 | val | deep_call_wing | medium | 100,624 | 29,408 | 29.23% | 0.1614 | 0.1615 |
| spy_phase1_random40_noiselow | 8 | val | deep_call_wing | short | 21,111 | 976 | 4.62% | 0.2532 | 0.2529 |
| spy_phase1_random40_noiselow | 8 | val | deep_put_wing | long | 258,994 | 102,872 | 39.72% | 0.1339 | 0.1336 |
| spy_phase1_random40_noiselow | 8 | val | deep_put_wing | medium | 304,173 | 120,859 | 39.73% | 0.1592 | 0.1586 |
| spy_phase1_random40_noiselow | 8 | val | deep_put_wing | short | 165,271 | 52,322 | 31.66% | 0.3122 | 0.3120 |
| spy_phase1_random40_noiselow | 8 | val | put_wing | long | 182,287 | 73,337 | 40.23% | 0.0275 | 0.0269 |
| spy_phase1_random40_noiselow | 8 | val | put_wing | medium | 282,586 | 112,944 | 39.97% | 0.0281 | 0.0266 |
| spy_phase1_random40_noiselow | 8 | val | put_wing | short | 350,916 | 135,688 | 38.67% | 0.1335 | 0.1326 |
| spy_phase1_random40_noiselow | 10 | test | atm | long | 145,965 | 58,069 | 39.78% | 0.0333 | 0.0332 |
| spy_phase1_random40_noiselow | 10 | test | atm | medium | 346,006 | 138,155 | 39.93% | 0.0192 | 0.0175 |
| spy_phase1_random40_noiselow | 10 | test | atm | short | 490,909 | 192,479 | 39.21% | 0.0467 | 0.0451 |
| spy_phase1_random40_noiselow | 10 | test | call_wing | long | 153,487 | 61,212 | 39.88% | 0.0920 | 0.0920 |
| spy_phase1_random40_noiselow | 10 | test | call_wing | medium | 203,692 | 80,821 | 39.68% | 0.1099 | 0.1098 |
| spy_phase1_random40_noiselow | 10 | test | call_wing | short | 145,334 | 34,690 | 23.87% | 0.1443 | 0.1442 |
| spy_phase1_random40_noiselow | 10 | test | deep_call_wing | long | 194,946 | 76,348 | 39.16% | 0.2548 | 0.2548 |
| spy_phase1_random40_noiselow | 10 | test | deep_call_wing | medium | 58,556 | 19,657 | 33.57% | 0.2464 | 0.2464 |
| spy_phase1_random40_noiselow | 10 | test | deep_call_wing | short | 20,305 | 413 | 2.03% | 0.2730 | 0.2733 |
| spy_phase1_random40_noiselow | 10 | test | deep_put_wing | long | 316,824 | 126,175 | 39.82% | 0.2465 | 0.2465 |
| spy_phase1_random40_noiselow | 10 | test | deep_put_wing | medium | 332,945 | 132,689 | 39.85% | 0.2411 | 0.2409 |
| spy_phase1_random40_noiselow | 10 | test | deep_put_wing | short | 154,446 | 46,400 | 30.04% | 0.3572 | 0.3572 |
| spy_phase1_random40_noiselow | 10 | test | put_wing | long | 196,169 | 78,976 | 40.26% | 0.0410 | 0.0408 |
| spy_phase1_random40_noiselow | 10 | test | put_wing | medium | 361,791 | 144,121 | 39.84% | 0.0415 | 0.0403 |
| spy_phase1_random40_noiselow | 10 | test | put_wing | short | 363,759 | 138,881 | 38.18% | 0.1780 | 0.1776 |
| spy_phase1_random40_noiselow | 10 | train | atm | long | 426,888 | 171,038 | 40.07% | 0.0375 | 0.0370 |
| spy_phase1_random40_noiselow | 10 | train | atm | medium | 575,252 | 231,097 | 40.17% | 0.0248 | 0.0234 |
| spy_phase1_random40_noiselow | 10 | train | atm | short | 655,715 | 255,623 | 38.98% | 0.0388 | 0.0368 |
| spy_phase1_random40_noiselow | 10 | train | call_wing | long | 439,488 | 176,255 | 40.10% | 0.0562 | 0.0562 |
| spy_phase1_random40_noiselow | 10 | train | call_wing | medium | 441,926 | 163,874 | 37.08% | 0.0718 | 0.0717 |
| spy_phase1_random40_noiselow | 10 | train | call_wing | short | 238,485 | 60,234 | 25.26% | 0.1169 | 0.1170 |
| spy_phase1_random40_noiselow | 10 | train | deep_call_wing | long | 321,436 | 113,139 | 35.20% | 0.1126 | 0.1127 |
| spy_phase1_random40_noiselow | 10 | train | deep_call_wing | medium | 164,464 | 27,207 | 16.54% | 0.1502 | 0.1505 |
| spy_phase1_random40_noiselow | 10 | train | deep_call_wing | short | 45,809 | 3,429 | 7.49% | 0.4084 | 0.4086 |
| spy_phase1_random40_noiselow | 10 | train | deep_put_wing | long | 736,461 | 293,537 | 39.86% | 0.1875 | 0.1876 |
| spy_phase1_random40_noiselow | 10 | train | deep_put_wing | medium | 791,285 | 291,893 | 36.89% | 0.1859 | 0.1857 |
| spy_phase1_random40_noiselow | 10 | train | deep_put_wing | short | 241,609 | 55,709 | 23.06% | 0.3267 | 0.3272 |
| spy_phase1_random40_noiselow | 10 | train | put_wing | long | 514,528 | 205,749 | 39.99% | 0.0482 | 0.0482 |
| spy_phase1_random40_noiselow | 10 | train | put_wing | medium | 600,304 | 239,938 | 39.97% | 0.0449 | 0.0440 |
| spy_phase1_random40_noiselow | 10 | train | put_wing | short | 476,917 | 176,683 | 37.05% | 0.1158 | 0.1151 |
| spy_phase1_random40_noiselow | 10 | val | atm | long | 155,320 | 61,748 | 39.76% | 0.0231 | 0.0222 |
| spy_phase1_random40_noiselow | 10 | val | atm | medium | 285,366 | 114,486 | 40.12% | 0.0183 | 0.0165 |
| spy_phase1_random40_noiselow | 10 | val | atm | short | 396,851 | 156,916 | 39.54% | 0.0363 | 0.0342 |
| spy_phase1_random40_noiselow | 10 | val | call_wing | long | 160,994 | 64,391 | 40.00% | 0.0386 | 0.0379 |
| spy_phase1_random40_noiselow | 10 | val | call_wing | medium | 243,488 | 95,985 | 39.42% | 0.0626 | 0.0621 |
| spy_phase1_random40_noiselow | 10 | val | call_wing | short | 253,946 | 69,761 | 27.47% | 0.1154 | 0.1151 |
| spy_phase1_random40_noiselow | 10 | val | deep_call_wing | long | 192,651 | 74,710 | 38.78% | 0.1344 | 0.1342 |
| spy_phase1_random40_noiselow | 10 | val | deep_call_wing | medium | 100,624 | 29,408 | 29.23% | 0.1614 | 0.1615 |
| spy_phase1_random40_noiselow | 10 | val | deep_call_wing | short | 21,111 | 976 | 4.62% | 0.2532 | 0.2529 |
| spy_phase1_random40_noiselow | 10 | val | deep_put_wing | long | 258,994 | 102,872 | 39.72% | 0.1339 | 0.1336 |
| spy_phase1_random40_noiselow | 10 | val | deep_put_wing | medium | 304,173 | 120,859 | 39.73% | 0.1592 | 0.1586 |
| spy_phase1_random40_noiselow | 10 | val | deep_put_wing | short | 165,271 | 52,322 | 31.66% | 0.3122 | 0.3120 |
| spy_phase1_random40_noiselow | 10 | val | put_wing | long | 182,287 | 73,337 | 40.23% | 0.0275 | 0.0269 |
| spy_phase1_random40_noiselow | 10 | val | put_wing | medium | 282,586 | 112,944 | 39.97% | 0.0281 | 0.0266 |
| spy_phase1_random40_noiselow | 10 | val | put_wing | short | 350,916 | 135,688 | 38.67% | 0.1335 | 0.1326 |
| spy_phase1_random40_noiselow | 12 | test | atm | long | 145,965 | 58,069 | 39.78% | 0.0333 | 0.0332 |
| spy_phase1_random40_noiselow | 12 | test | atm | medium | 346,006 | 138,155 | 39.93% | 0.0192 | 0.0175 |
| spy_phase1_random40_noiselow | 12 | test | atm | short | 490,909 | 192,479 | 39.21% | 0.0467 | 0.0451 |
| spy_phase1_random40_noiselow | 12 | test | call_wing | long | 153,487 | 61,212 | 39.88% | 0.0920 | 0.0920 |
| spy_phase1_random40_noiselow | 12 | test | call_wing | medium | 203,692 | 80,821 | 39.68% | 0.1099 | 0.1098 |
| spy_phase1_random40_noiselow | 12 | test | call_wing | short | 145,334 | 34,690 | 23.87% | 0.1443 | 0.1442 |
| spy_phase1_random40_noiselow | 12 | test | deep_call_wing | long | 194,946 | 76,348 | 39.16% | 0.2548 | 0.2548 |
| spy_phase1_random40_noiselow | 12 | test | deep_call_wing | medium | 58,556 | 19,657 | 33.57% | 0.2464 | 0.2464 |
| spy_phase1_random40_noiselow | 12 | test | deep_call_wing | short | 20,305 | 413 | 2.03% | 0.2730 | 0.2733 |
| spy_phase1_random40_noiselow | 12 | test | deep_put_wing | long | 316,824 | 126,175 | 39.82% | 0.2465 | 0.2465 |
| spy_phase1_random40_noiselow | 12 | test | deep_put_wing | medium | 332,945 | 132,689 | 39.85% | 0.2411 | 0.2409 |
| spy_phase1_random40_noiselow | 12 | test | deep_put_wing | short | 154,446 | 46,400 | 30.04% | 0.3572 | 0.3572 |
| spy_phase1_random40_noiselow | 12 | test | put_wing | long | 196,169 | 78,976 | 40.26% | 0.0410 | 0.0408 |
| spy_phase1_random40_noiselow | 12 | test | put_wing | medium | 361,791 | 144,121 | 39.84% | 0.0415 | 0.0403 |
| spy_phase1_random40_noiselow | 12 | test | put_wing | short | 363,759 | 138,881 | 38.18% | 0.1780 | 0.1776 |
| spy_phase1_random40_noiselow | 12 | train | atm | long | 426,888 | 171,038 | 40.07% | 0.0375 | 0.0370 |
| spy_phase1_random40_noiselow | 12 | train | atm | medium | 575,252 | 231,097 | 40.17% | 0.0248 | 0.0234 |
| spy_phase1_random40_noiselow | 12 | train | atm | short | 655,715 | 255,623 | 38.98% | 0.0388 | 0.0368 |
| spy_phase1_random40_noiselow | 12 | train | call_wing | long | 439,488 | 176,255 | 40.10% | 0.0562 | 0.0562 |
| spy_phase1_random40_noiselow | 12 | train | call_wing | medium | 441,926 | 163,874 | 37.08% | 0.0718 | 0.0717 |
| spy_phase1_random40_noiselow | 12 | train | call_wing | short | 238,485 | 60,234 | 25.26% | 0.1169 | 0.1170 |
| spy_phase1_random40_noiselow | 12 | train | deep_call_wing | long | 321,436 | 113,139 | 35.20% | 0.1126 | 0.1127 |
| spy_phase1_random40_noiselow | 12 | train | deep_call_wing | medium | 164,464 | 27,207 | 16.54% | 0.1502 | 0.1505 |
| spy_phase1_random40_noiselow | 12 | train | deep_call_wing | short | 45,809 | 3,429 | 7.49% | 0.4084 | 0.4086 |
| spy_phase1_random40_noiselow | 12 | train | deep_put_wing | long | 736,461 | 293,537 | 39.86% | 0.1875 | 0.1876 |
| spy_phase1_random40_noiselow | 12 | train | deep_put_wing | medium | 791,285 | 291,893 | 36.89% | 0.1859 | 0.1857 |
| spy_phase1_random40_noiselow | 12 | train | deep_put_wing | short | 241,609 | 55,709 | 23.06% | 0.3267 | 0.3272 |
| spy_phase1_random40_noiselow | 12 | train | put_wing | long | 514,528 | 205,749 | 39.99% | 0.0482 | 0.0482 |
| spy_phase1_random40_noiselow | 12 | train | put_wing | medium | 600,304 | 239,938 | 39.97% | 0.0449 | 0.0440 |
| spy_phase1_random40_noiselow | 12 | train | put_wing | short | 476,917 | 176,683 | 37.05% | 0.1158 | 0.1151 |
| spy_phase1_random40_noiselow | 12 | val | atm | long | 155,320 | 61,748 | 39.76% | 0.0231 | 0.0222 |
| spy_phase1_random40_noiselow | 12 | val | atm | medium | 285,366 | 114,486 | 40.12% | 0.0183 | 0.0165 |
| spy_phase1_random40_noiselow | 12 | val | atm | short | 396,851 | 156,916 | 39.54% | 0.0363 | 0.0342 |
| spy_phase1_random40_noiselow | 12 | val | call_wing | long | 160,994 | 64,391 | 40.00% | 0.0386 | 0.0379 |
| spy_phase1_random40_noiselow | 12 | val | call_wing | medium | 243,488 | 95,985 | 39.42% | 0.0626 | 0.0621 |
| spy_phase1_random40_noiselow | 12 | val | call_wing | short | 253,946 | 69,761 | 27.47% | 0.1154 | 0.1151 |
| spy_phase1_random40_noiselow | 12 | val | deep_call_wing | long | 192,651 | 74,710 | 38.78% | 0.1344 | 0.1342 |
| spy_phase1_random40_noiselow | 12 | val | deep_call_wing | medium | 100,624 | 29,408 | 29.23% | 0.1614 | 0.1615 |
| spy_phase1_random40_noiselow | 12 | val | deep_call_wing | short | 21,111 | 976 | 4.62% | 0.2532 | 0.2529 |
| spy_phase1_random40_noiselow | 12 | val | deep_put_wing | long | 258,994 | 102,872 | 39.72% | 0.1339 | 0.1336 |
| spy_phase1_random40_noiselow | 12 | val | deep_put_wing | medium | 304,173 | 120,859 | 39.73% | 0.1592 | 0.1586 |
| spy_phase1_random40_noiselow | 12 | val | deep_put_wing | short | 165,271 | 52,322 | 31.66% | 0.3122 | 0.3120 |
| spy_phase1_random40_noiselow | 12 | val | put_wing | long | 182,287 | 73,337 | 40.23% | 0.0275 | 0.0269 |
| spy_phase1_random40_noiselow | 12 | val | put_wing | medium | 282,586 | 112,944 | 39.97% | 0.0281 | 0.0266 |
| spy_phase1_random40_noiselow | 12 | val | put_wing | short | 350,916 | 135,688 | 38.67% | 0.1335 | 0.1326 |

### 2.4 Sparse-region density sensitivity

Nearest-observed L2 distance from each held-out point to the observed set on the same date, computed three ways: **naive** (raw observed set), **dedup_obs** (collapse observed by rounded coordinate, keep one representative), **exclude_self_dup** (drop observed rows sharing the rounded coordinate of the hidden row).

| Benchmark | dp | Mode | n hidden | n zero-dist | % zero | q25 | q50 | q75 | q95 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| spy_phase1_random40_noiselow | 10 | dedup_obs | 13,510,279 | 5,060,894 | 37.46% | 0.0000 | 0.0070 | 0.0106 | 0.0263 |
| spy_phase1_random40_noiselow | 10 | exclude_self_dup | 13,510,279 | 0 | 0.00% | 0.0071 | 0.0086 | 0.0130 | 0.0299 |
| spy_phase1_random40_noiselow | 10 | naive | 13,510,279 | 5,060,894 | 37.46% | 0.0000 | 0.0070 | 0.0106 | 0.0263 |

## 3. Interpretation

The IV surface is defined as `IV(log_moneyness, tau)`. Including both calls and puts at the same `(date, expiration, strike)` violates the single-valued-function assumption: at the same model coordinate the loss sees two distinct labels, and the RBF kernel must average over them. Put-call parity says the *price* structure is consistent, but the *implied volatilities* reported by the data source diverge whenever bid/ask sits asymmetrically around the parity-implied mid (which is the usual case for non-ATM strikes). Section 2.2 quantifies that divergence.

Section 2.3 then quantifies how often held-out targets in the benchmark have an observed exact twin on the same date. Whenever they do, the sparse-region analysis's nearest-observed distance is *forced* to zero — not because the local neighbourhood is dense, but because the held-out point is literally one leg of a call-put pair whose other leg leaked into context.


## 4. Recommendation

Duplicate coordinates are severe. The current strict surface table is not a single-valued IV surface; it is a quote table. **Recommended action before any sparse-region run:** ship Option A (true OTM surface, puts for K<S / calls for K>S, tie-break at ATM) as a new derived parquet, e.g. `spy_surface_points_strict_otm.parquet`, leave the existing file untouched, regenerate the benchmark from the OTM file, and re-run the sparse-region experiment on it. Option D (dedup-aware density) is a stopgap if Option A is too costly to ship before the experiment, but the experiment then needs to be reported with the duplicate-affected slice excluded.

## 5. Sparse-region experiment status

Sparse-region ANP-vs-RBF needs data reconstruction (Option A) before it can be claimed as a clean test of the hypothesis.
