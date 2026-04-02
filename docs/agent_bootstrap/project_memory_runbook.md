# Project Memory Reviewer — Runbook

> Operating sequence for future reviewer runs.

---
created_at: 2026-04-02T00:00:00-04:00
last_updated_at: 2026-04-02T01:46:00-04:00
---

## Prerequisites

Before running the reviewer, build a review packet:

```bash
python scripts/build_project_memory_review_packet.py
```

This writes:
- `reports/project_memory/latest_review_packet.md` — human-readable context summary
- `reports/project_memory/latest_review_packet_fingerprint.json` — machine-readable fingerprint

---

## Operating sequence

### Step 0 — Check reviewer state (idempotence gate)

Before doing any work, read:

1. `docs/agent_bootstrap/reviewer_state.json`
2. `reports/project_memory/latest_review_packet_fingerprint.json` (if available)

Check for repeat-run conditions:

- If `last_processed_git_head` matches current git HEAD **AND** the diff signature matches `last_processed_diff_signature` **AND** there are no new untracked project-memory files → **no-op**. Report: "No material project-memory updates required."
- If `bootstrap_completed` is true and the user asked for a bootstrap → do not recreate files. Report that bootstrap is already installed and offer to review/update existing files minimally.
- If `baseline_completed` is true and the user asked for a baseline review → check if git HEAD has advanced since `baseline_completed_at`. If HEAD is the same and diff signature matches, **no-op**.

If none of the above apply, proceed to Step 1.

### Step 1 — Gather evidence

1. Read `reports/project_memory/latest_review_packet.md` (if available).
2. Run `git status` to see current working tree state.
3. Run `git diff --stat` to see what files changed (staged and unstaged).
4. Run `git log --oneline -20` to see recent commit history.
5. If the diff is non-trivial, read the actual changed files to understand what happened.

### Step 2 — Inspect relevant evidence files

Based on what changed, read the relevant evidence:

- If source code changed → read the changed source files
- If configs changed → read the changed configs
- If notebooks changed → note the notebook, check for new outputs
- If training/experiment outputs exist → check `logs/`, `plots/`, `reports/`
- If test files changed → read the test changes

### Step 3 — Classify semantic change events

For each meaningful change, classify it using the taxonomy in `docs/agent_bootstrap/change_event_taxonomy.md`:

- Code progress
- Routine experiment run
- Failed run / mistake
- Architecture decision
- Roadmap change
- Milestone completion
- Infrastructure / workflow issue

One commit or diff may contain multiple change events.

### Step 4 — Determine which memory files to update

Consult `docs/agent_bootstrap/project_memory_update_policy.md` to determine:

- Which file(s) to update for each classified event
- What update mode to use (append, curated edit, new numbered doc)
- What timestamp format to use

### Step 5 — Check for duplicates before writing

Before writing to any memory file, check whether a semantically equivalent entry or document already exists. Use these signals:

**For append-only logs** (`progress_log.md`, `experiment_journal.md`):
- Scan the last 5–10 entries for matching timestamps, commit hashes, or descriptions.
- If an entry already covers the same event (same commit hash, same milestone, same experiment), do not append a duplicate.
- If the existing entry is incomplete or inaccurate, update it in place rather than appending a new one.

**For numbered docs** (`docs/retrospectives/`, `docs/decisions/`):
- Check `reviewer_state.json` for known IDs.
- Search existing files for the same `event_at`, related files, or root cause.
- If a retrospective or ADR already covers the same underlying event, update the existing doc if needed rather than creating a duplicate.

**When uncertain**: report the ambiguity to the user rather than risking a duplicate. A missing entry can be added later; a duplicate is harder to clean up.

### Step 6 — Execute updates

For each file that needs updating:

1. Read the current file content first.
2. Match the existing formatting style.
3. Apply the minimum change needed.
4. Add correct timestamps.
5. Do not touch files that are not justified by the evidence.

### Step 7 — Update reviewer state

After making material changes (or confirming no-op), update:

- `docs/agent_bootstrap/reviewer_state.json` — update `last_review_run_at`, `last_processed_git_head`, `last_processed_diff_signature`, `last_review_run_type`, and any newly created IDs.
- `docs/agent_bootstrap/reviewer_state.md` — update the human-readable summary to match.

If no material changes were made, still update `last_review_run_at` and `last_processed_git_head` to record that the state was inspected.

### Step 8 — Summarize

Report to the user:

1. What change events were detected
2. What memory files were updated (and how)
3. What files were intentionally not touched
4. Any ambiguity or uncertainty about what happened
5. Suggested follow-up actions if applicable
6. Whether this was a no-op (and why)

---

## Reviewer contract

Future reviewer runs MUST:

1. **Be evidence-based.** Every claim must trace to observable evidence (git diff, file content, log output). Never infer outcomes from intentions or plans.

2. **Avoid hallucinating outcomes.** If a training run's results are not visible in the evidence, do not claim what the results were. Say "results not available in evidence" instead.

3. **Distinguish completed work from intended work.** "Implemented X" means the code exists and is functional. "Plans to implement X" means it does not yet exist. Never conflate these.

4. **Distinguish mistakes from hypotheses.** A mistake is something that went wrong and caused a problem. A hypothesis that didn't pan out in an experiment is not a mistake — it's a normal experiment outcome. Only create retrospectives for genuine failures, not for routine negative results.

5. **Preserve the repo's existing documentation style where practical.** Match heading levels, section structure, bullet formatting, and tone of the file being edited. Do not impose a foreign style.

6. **Prefer minimal, surgical edits.** Update the specific section that needs updating. Do not rewrite surrounding content for consistency or style improvement unless explicitly asked.

7. **Create new numbered docs for major retrospectives and ADRs** rather than overloading unrelated files. Use `docs/retrospectives/NNNN_short_name.md` and `docs/decisions/NNNN_short_name.md`.

8. **Never auto-edit source code, tests, scripts, notebooks, or configs.** The reviewer system manages project memory, not implementation.

9. **Never fabricate timestamps.** If the event time cannot be reliably inferred from evidence, omit it or mark it as unknown.

10. **Never over-claim.** If you are unsure whether something constitutes a milestone, a mistake, or a decision, err on the side of not recording it. A missing entry can be added later; a wrong entry is harder to undo.

11. **Never create duplicates.** Always check existing entries before writing. Use Step 5 (duplicate check) before every write.

12. **Update reviewer state after every run.** Even no-op runs should update `last_review_run_at` and `last_processed_git_head` in `reviewer_state.json`.

---

## Idempotence rules by run type

### Bootstrap run

A bootstrap installs the reviewer system scaffolding.

- **If `reviewer_state.json` exists and `bootstrap_completed` is true:** do not recreate bootstrap files. Instead, review existing files and apply minimal updates if the user explicitly requests changes.
- **If bootstrap files are missing or corrupted:** recreate only the missing files. Do not overwrite files that already exist and are correct.
- **Running the bootstrap prompt twice produces no new files** — the reviewer checks state, confirms bootstrap is done, and reports a no-op.

### Baseline review run

A baseline review cross-references git history against existing project-memory files to backfill gaps.

- **If `reviewer_state.json` exists and `baseline_completed` is true AND git HEAD matches `last_processed_git_head`:** no-op. Report that baseline was already completed for this repo state.
- **If git HEAD has advanced since the baseline was completed:** this is not a baseline re-run — it is an ongoing review. Use the ongoing flow instead.
- **Running the baseline prompt twice on the same repo state produces no changes** — the reviewer detects the match and reports a no-op.

### Ongoing review run

An ongoing review processes new changes since the last review.

- **If git HEAD matches `last_processed_git_head` AND diff signature matches `last_processed_diff_signature`:** no-op. Report "No material project-memory updates required."
- **If git HEAD has advanced but diff signature is unchanged** (e.g., only the review packet files changed): still check whether the new commits contain semantic changes worth logging. If not, no-op.
- **If the reviewer would append an entry identical to the last entry in the target file:** do not append. This is a duplicate.

---

## No-op is a valid result

The reviewer is explicitly allowed — and expected — to produce:

> **No material project-memory updates required.**

This is the correct output when:

- There are no meaningful changes since the last processed state.
- The same run was already processed (same git HEAD + diff signature).
- Evidence is insufficient to justify any update.
- A repeat invocation would only create duplicates.

A no-op run should still update `last_review_run_at` in `reviewer_state.json` to record that the state was inspected.

---

## What happens on repeat runs (safety reference)

| Scenario | Expected behavior |
|---|---|
| Bootstrap run when bootstrap is already done | No-op. Reports bootstrap already installed. |
| Baseline run when baseline was already completed for this HEAD | No-op. Reports baseline already completed. |
| Ongoing run with no new commits or diffs | No-op. Reports no material updates required. |
| Ongoing run with new commits but no semantic change | No-op or minimal update. Only records truly new events. |
| Ongoing run that would duplicate an existing progress entry | Skips the duplicate. Reports why. |
| Ongoing run that would duplicate an existing retrospective | Skips creation. Reports the existing retro covers the event. |
| Any run where the reviewer is uncertain | Reports ambiguity to the user. Does not write. |
