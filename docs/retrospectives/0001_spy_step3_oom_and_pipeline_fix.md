# Retrospective 0001: SPY Step 3 OOM and Pipeline Design Fix

## Date

2026-03-31

## Related files

- `src/data/03_build_spy_surface_table.py`
- `docs/data_assumptions_and_cleaning.md`
- `docs/decisions/0002_phase1_scope_freeze.md`

---

## 1. What happened

SPY data ingestion (step 1) and schema inspection (step 2) both completed successfully on RunPod. Step 1 downloaded 24,681,665 option rows and 4,529 underlying price rows. Step 2 confirmed all required columns were present with minimal quality issues.

Step 3 — the build script that joins options with underlying data, derives surface coordinates, applies cleaning, and writes processed Parquet — failed. The RunPod Pod became unresponsive and stopped accepting SSH connections. Multiple concurrent SSH sessions running the build script compounded the problem.

## 2. Why this was a mistake

The build script (`src/data/03_build_spy_surface_table.py`) used an eager, single-pass pandas workflow: load the full ~24.7M-row options table into memory, load the underlying table, merge them, derive all columns, apply all cleaning, sort, and write Parquet in one shot.

A 632 MB Parquet file expands to several gigabytes in a pandas DataFrame. After the join and column derivation, the in-memory footprint exceeded available system RAM on the RunPod Pod, causing an out-of-memory condition.

This was not a data-quality mistake. The schema was correct, the cleaning logic was sound, and the thresholds were reasonable. It was a pipeline execution and data engineering design mistake.

## 3. Root cause

- The script treated a 24.7M-row dataset as if it were a small notebook-scale table.
- We validated schema correctness but did not validate execution scalability.
- We moved directly from successful ingestion and inspection to a full-scale build without first testing on a single year or partition.
- The mental model was "load everything into a DataFrame and transform it," when the correct model for this data size is a partitioned transform pipeline.

## 4. What we learned

- Schema validation does not imply execution readiness. A pipeline step can be logically correct but operationally infeasible.
- A 600 MB Parquet file is not a small file. The in-memory expansion factor for wide DataFrames with mixed types is roughly 3–5×.
- Always test a pipeline on one partition before running at full scale.
- Multiple concurrent SSH sessions running memory-heavy jobs will compound OOM risk and can render the Pod unrecoverable without a restart.

## 5. Improvement plan

The refactored Step 3 must:

1. **Process partition-by-partition** — iterate over years (or date ranges) rather than loading the full table at once.
2. **Prune columns early** — select only required and optional columns immediately after reading each partition, before joining.
3. **Use memory-safe joins** — join each partition with the small underlying table (4,529 rows), not the reverse.
4. **Write intermediate Parquet outputs** — write each processed partition to disk before moving to the next.
5. **Concatenate at the end** — optionally merge partitions into a single file after all partitions succeed, or keep them partitioned.
6. **Add progress logging** — print partition name, row counts, and elapsed time for each step.
7. **Test on one year first** — validate end-to-end on a single year (e.g., 2024) before running the full 2008–2025 range.

## 6. Updated implementation plan

```
for each year in [2008, 2009, ..., 2025]:
    1. read options rows for that year (filter on date column)
    2. select only needed columns
    3. join with underlying on date
    4. compute derived columns (mid, tau, log_moneyness, spot)
    5. apply hard cleaning drops
    6. add quality flags
    7. write partition to data_processed/spy/spy_surface_points_{year}.parquet
    8. log row counts and timing

after all years:
    - concatenate partitions into spy_surface_points.parquet (if memory permits, otherwise keep partitioned)
    - build strict subset from the concatenated or partitioned output
    - write spy_surface_points_strict.parquet
    - generate build report
```

## 7. Why we are not doing "optimal cleaning" first

At this stage of Phase 1, we intentionally avoid aggressive or fine-tuned filtering:

- We do not yet have downstream modeling evidence to set optimal thresholds. Thresholds should emerge from experimentation, not from guesses.
- Aggressive filtering too early removes real market structure (zero-bid deep OTM options, wide spreads during volatile sessions, etc.) that may be informative for understanding model behavior.
- Over-cleaning makes debugging harder. When a model produces unexpected results, it helps to have the full picture available for inspection.
- It slows iteration. Every cleaning decision becomes a bottleneck if it requires justification before the first model run.

## 8. Why we also want a stricter subset

A completely unfiltered dataset creates noisy, unstable early experiments. For the first baseline and neural model runs, we want a subset where:

- Bid and ask are both positive (no zero-bid deep OTM noise)
- Mid price is nonzero
- IV, tau, and log-moneyness are within reasonable documented bounds
- The data is dense enough to form a plausible surface

This stricter subset exists to make early modeling iterations faster and more interpretable. It is not the final production filter.

## 9. Current decision

Produce two outputs from Step 3:

| Output | Purpose | Filtering |
|--------|---------|-----------|
| `spy_surface_points.parquet` | Broad processed table | Hard drops only (nulls, negatives, crossed quotes, IV ≤ 0, tau ≤ 0). Quality flags attached but not used for exclusion. |
| `spy_surface_points_strict.parquet` | Strict modeling subset | Additional filters: positive bid/ask, nonzero mid, bounded IV, bounded tau, bounded log-moneyness. Thresholds documented in `src/data/config.py`. |

The broad table supports auditability, debugging, and truthfulness to market structure. The strict subset supports cleaner first-pass experiments and more stable baseline runs.

## 10. Immediate next action

Refactor `src/data/03_build_spy_surface_table.py` to use a year-by-year partitioned workflow. Test on one year first. Then run full scale.

## 11. One-sentence summary

Step 3 failed because we used a notebook-style eager DataFrame workflow on a 24.7M-row dataset; the fix is a partitioned, memory-safe pipeline that we test on one year before running at full scale.
