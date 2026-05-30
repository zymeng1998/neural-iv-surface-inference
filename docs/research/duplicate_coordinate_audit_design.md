# Duplicate-coordinate audit — design + pre-run static evidence

---
created_at: 2026-05-29T00:00:00-04:00
last_updated_at: 2026-05-29T00:00:00-04:00
status: proposed
type: data_audit_design
relates_to:
  - docs/research/sparse_region_anp_vs_rbf_design.md
  - src/data/03_build_spy_surface_table.py
  - src/data/04_build_benchmark_tasks.py
  - src/neural_iv_surface_inference/data/conditional_loaders.py
  - src/neural_iv_surface_inference/models/interpolation.py
---

> Companion doc to [`scripts/audit_duplicate_coordinates.py`](../../scripts/audit_duplicate_coordinates.py).
> The script generates `docs/research/duplicate_coordinate_audit.md` with
> the *numerical* findings on Pod data. This file captures the *structural*
> finding from a code-only audit and locks the decision matrix in advance,
> so the Pod run reduces to fill-in-the-blanks plus a final verdict.

## 1. The structural defect (no data needed to find)

The conditional model and the RBF baseline both treat IV as a single-valued
function of `(log_moneyness, tau)`:

- `src/neural_iv_surface_inference/data/conditional_loaders.py:27`
  ```python
  _CONTEXT_FEATURES = ("log_moneyness", "tau", "implied_volatility")
  _QUERY_FEATURES   = ("log_moneyness", "tau")
  ```
- `src/neural_iv_surface_inference/models/interpolation.py:45`
  ```python
  coords_obs = obs[["log_moneyness", "tau"]].values
  iv_obs     = obs["implied_volatility"].values
  ```

`type` (call / put) is **not** a model feature.

The strict surface table, however, retains both legs of every contract.
[`src/data/03_build_spy_surface_table.py`](../../src/data/03_build_spy_surface_table.py)
applies the following cleaning gates and nothing else:

| Gate | Code reference |
|---|---|
| Drop rows with null `date / expiration / strike / type / iv / close` | lines 159–161 |
| Drop crossed quotes (`bid < 0`, `ask < bid`) | lines 163–165 |
| Drop `iv ∉ (0, 5]`, `tau ∉ (0, 3]` | lines 167–172 |
| Strict subset: tighter IV / tau / log-moneyness windows | `build_strict_subset` |

There is **no** `drop_duplicates(["date", "expiration", "strike"])`, no
`groupby(["date", "expiration", "strike"]).agg(...)`, and no OTM filter
(`K < S ⇒ put`, `K > S ⇒ call`). Both the call leg and the put leg of
the same contract pass the gates independently whenever their bids and
IVs are individually finite — i.e. **for nearly every listed contract**
(both legs of every American option have valid bid/ask quotes during
RTH; AlphaVantage's `HISTORICAL_OPTIONS` returns both rows).

`src/data/04_build_benchmark_tasks.py` consumes the strict file unchanged
and only adds `observed`, `iv_clean`, `noise_sigma`, `split` columns. The
duplicate structure is preserved into every benchmark variant.

**Consequence at the model coordinate `(log_m, tau)`:** on every date,
the same `(K, T)` contract emits two rows with two different IVs (the
call IV and the put IV reported by AV). Whenever one leg is masked
`observed=True` and the other `observed=False`, the held-out target sits
at L2-distance **exactly zero** from an observed point, regardless of
whether the local surface neighbourhood is actually dense. This is
structural call-put leakage, not a benchmark accident.

## 2. What the script measures

The audit script is fully streaming (per-date, chunked PyArrow). On Pod
it should run in a few minutes per benchmark. Numerical fields it fills:

### 2.1 Strict surface (`duplicate_summary.csv` + `duplicate_iv_dispersion.csv`)

For each of the keys
- `(date, expiration, strike)` — contract key
- `(date, round(log_m, d), round(tau, d))` for `d ∈ {8, 10, 12}` — coord key

it reports:
- total rows
- rows in duplicate groups, absolute + percentage
- number of duplicate groups
- size histogram (size=2, size=3, size≥4)
- call-put-mixed groups vs same-type groups
- IV-range = `max(IV) − min(IV)` per group: `n`, mean, p50, p90, p95, p99, max
  — split by call-put-mix vs same-type

### 2.2 Benchmark splits (`observed_hidden_leakage.csv`)

For each benchmark, for each `(split × moneyness_bucket × maturity_bucket)`
cell, and for each of `d ∈ {8, 10, 12}`:
- `n_hidden_rows` — total held-out points in the cell
- `n_hidden_with_obs_twin` — held-out points with at least one observed
  row at the exact rounded coordinate on the same date
- `pct_hidden_with_twin`
- `twin_mae` — MAE of the trivial baseline that predicts the mean
  observed IV at the exact coordinate (so we can compare the call-put
  duplicate's twin-IV vs the held-out put-leg's `iv_clean`)
- `twin_iv_range_mean` — average dispersion at the leaked coordinate

Moneyness buckets mirror `src/neural_iv_surface_inference/training/eval.py`:
`deep_put_wing < -0.20`, `put_wing [-0.20, -0.05)`, `atm [-0.05, 0.05)`,
`call_wing [0.05, 0.20)`, `deep_call_wing ≥ 0.20`.

Maturity buckets: `short < 30d`, `medium [30, 180)d`, `long ≥ 180d`.

### 2.3 Sparse-region density sensitivity (`sparse_density_sensitivity.csv`)

For each benchmark at `d = 10`, per held-out row, the nearest-observed
L2 distance is computed three ways:

| Mode | Definition |
|---|---|
| `naive` | All observed rows. (Current `sparse_region_anp_vs_rbf` design.) |
| `dedup_obs` | Collapse observed by rounded `(log_m, tau)`, keep one rep. |
| `exclude_self_dup` | Drop observed rows that share a rounded coord with the hidden row before computing nearest. |

Reported per (benchmark, decimals, mode):
- `n_total_hidden`, `n_zero_distance`, `pct_zero_distance`
- `n_hidden_with_exact_obs_dup` (naive mode only)
- distance quantiles q25, q50, q75, q95

The Q1 (densest) bucket in the sparse-region design's stratification is
defined by smallest nearest-observed distance — so any inflation of
zero-distance rows under `naive` directly contaminates Q1.

## 3. Decision matrix (locked in advance)

Verdict thresholds, applied to the contract-key dup share and the
10-decimal coord-key dup share in `duplicate_summary.csv`:

| Verdict | Threshold | Action before sparse-region ANP-vs-RBF |
|---|---|---|
| Negligible | both shares < 0.5 % | Run as-is. Include a one-line dedup sanity check in the report. |
| Moderate | contract share < 10 % AND coord share < 15 % | Run **Option D** only: dedup-aware density (`exclude_self_dup` mode) for all density-stratified MAE numbers. Report sensitivity vs naive. Do NOT rebuild data. |
| Severe | either share ≥ 10 % / 15 % | Defer the experiment. Ship **Option A** first: derive `spy_surface_points_strict_otm.parquet` (puts for `K<S`, calls for `K>S`, ATM tie-broken by mid-IV or by side with smaller spread), regenerate the benchmark from it, then re-run the experiment on the OTM benchmark. Keep the original strict file and benchmarks untouched. |

### Why Option A is preferred over Option B (aggregation) in the severe case

Option B (median or quality-weighted average of the call IV and put IV at
each `(date, K, T)`) collapses the two-leg structure but **invents** a
surface point that no market participant ever quoted. The OTM convention
(Option A) is what every commercial vol-surface vendor uses (CBOE VIX
methodology, Bloomberg `OVME`, OptionMetrics IvyDB) precisely because OTM
options carry the time value that defines the smile; the ITM leg is
dominated by intrinsic value and its IV is more sensitive to discrete
strike grid and to early-exercise approximation noise. Aggregation also
destroys the dispersion signal that 2.2 captures — and that dispersion
is the strongest direct measure of label noise in the dataset.

### Why we do not pick Option C (`type` as a feature)

Option C reframes the task from *surface inference* to *raw-quote
inference*. The Phase 3 framing ([ADR 0004](../decisions/0004_phase3_accuracy_push_framing.md))
is explicitly about modelling a single-valued vol surface and beating
RBF on that. Adding `type` would let the model memorise the call/put
gap, inflate apparent accuracy, and break the comparability to RBF (RBF
on raw quotes would need separate call and put surfaces or the same
`type` flag, at which point the comparison stops being apples-to-apples).

## 4. Execution

### Local (smoke test, already passing)

```bash
python3 -m pytest tests/test_audit_duplicate_coordinates.py -v
```

3 tests cover: contract-key dup counting, coordinate-key dup counting,
benchmark observed-hidden leakage detection on a hand-built fixture.

### Pod (real numbers)

```bash
ssh root@<pod-ip> -p <port> -i ~/.ssh/id_ed25519
cd /workspace/neural_iv_surface_inference
python scripts/audit_duplicate_coordinates.py \
  --strict   data_processed/spy/spy_surface_points_strict.parquet \
  --benchmark data_processed/spy/benchmarks/spy_phase1_random40_noiselow.parquet \
  --output-dir artifacts/audits/duplicate_coordinates \
  --report     docs/research/duplicate_coordinate_audit.md \
  > logs/audit_duplicate_coordinates.log 2>&1
tail -30 logs/audit_duplicate_coordinates.log
```

Estimated wall time: ~5 min for the strict file + ~2 min per benchmark
on an RTX-A4500 Pod (CPU-bound, single-process). Peak memory: bounded
by the largest single date's row count (~10k), well under 1 GB.

Pull the four CSV files plus the regenerated markdown back to local:

```bash
scp -P <port> -i ~/.ssh/id_ed25519 \
  root@<pod-ip>:/workspace/.../artifacts/audits/duplicate_coordinates/*.csv \
  artifacts/audits/duplicate_coordinates/
scp -P <port> -i ~/.ssh/id_ed25519 \
  root@<pod-ip>:/workspace/.../docs/research/duplicate_coordinate_audit.md \
  docs/research/
```

The four CSVs are small (one-line-per-cell summaries, ≤ a few hundred
KB) and safe to commit.

## 5. Non-goals for this audit pass

- No changes to `src/data/03_build_spy_surface_table.py` or
  `04_build_benchmark_tasks.py`.
- No new benchmark variant.
- No re-training. No model code touched.
- Option A or Option D implementation is a **follow-up** decided by the
  Pod-run verdict; specs land separately if needed.
