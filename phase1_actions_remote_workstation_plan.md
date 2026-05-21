# Phase 1 Actions — Remote Working Station First

_Last updated: 2026-04-03_

## 0. What this document is

This is the execution plan for **Phase 1** of the ML × Finance implied-volatility surface project.

It is **not** a resume summary and **not** an interview script.
It is a practical action plan for getting from:

- no remote environment,
- no data pipeline,
- no model,

to:

- a working remote development setup,
- one real-data pipeline,
- one neural baseline,
- one simple baseline,
- one vendor-style reference,
- first plots/tables/results.

---

# 1. Phase 1 objective

## Primary objective
Turn the project into a **real-data, benchmarkable neural inference prototype**.

## Minimum Phase 1 end state
By the end of Phase 1, we want:

1. A stable remote working station
2. One real EOD options dataset running end-to-end
3. One liquid underlying (start with **SPY**)
4. One simple non-neural baseline
5. One PyTorch neural baseline
6. One vendor-style reference stream
7. A small result package:
   - reference surface plot
   - sparse/noisy input plot
   - reconstructed surface plot
   - baseline vs neural comparison plot
   - error vs sparsity plot
   - one summary table

## Current Phase 1 status (alignment snapshot)

Completed from repository evidence:
- Remote workstation workflow is operational on RunPod (SSH + persistent workspace + Git workflow documented).
- SPY EOD data pipeline is implemented and run through strict surface generation (S1 complete).
- Benchmark task construction is implemented and run (S2 complete).
- Benchmark parity check is complete on RunPod: strict rows and all 11 benchmark files match expected row counts.
- Interpolation and masked MLP baselines are implemented with tests (S3.1 and S3.2 complete in code).
- Core evaluation metrics and regional diagnostics are implemented (S4.1 and S4.2 complete in code).

Remaining Phase 1 gaps:
- S3.3 vendor-style reference alignment is not yet implemented.
- S4.3 result artifact package (plots + summary table + memo) is not yet completed on real benchmark runs.

Immediate execution order to finish Phase 1:
1. Run real-data Step 5 baselines on RunPod (primary benchmark + small regime sweep).
2. Produce S4.3 artifacts from those runs.
3. Resolve S3.3 either by initial implementation or by documenting a concrete external dependency block with next action owner.

---

# 2. High-level action map

## Action group A — Establish remote working station
Goal: create a stable environment for code, notebooks, data, and backups.

## Action group B — Lock Phase 1 scope
Goal: reduce project surface area and avoid overbuilding.

## Action group C — Build data pipeline
Goal: get one real underlying, one date range, one clean dataset.

## Action group D — Define benchmark task
Goal: create sparse/noisy/irregular observation task from real chains.

## Action group E — Implement baselines
Goal: at least one simple baseline and one neural baseline.

## Action group F — Produce first results
Goal: get plots, metrics, and a narrative.

---

# 3. Action group A — Remote working station

## Target architecture
- **RunPod Secure Cloud Pod** = main remote compute machine
- **RunPod Network Volume** = persistent working storage
- **Cyberduck over SFTP** = GUI file transfer
- **AWS S3** = cold backup/archive
- Optional but recommended: **VS Code / Cursor Remote-SSH** for actual editing

## Important implementation note
RunPod's **basic proxied SSH does not support SCP or SFTP**. If you want Cyberduck SFTP or VS Code/Cursor direct SSH, you need a Pod/template that supports **SSH over exposed TCP / direct TCP port 22**. If port settings change, the Pod may restart, so keep important data on the persistent volume (`/workspace` or attached network volume mount), not only on ephemeral disk. citeturn452509search0turn452509search4turn452509search1

---

## A1. Create required accounts

### Checklist
- [ ] RunPod account created and billing added
- [ ] AWS account created and billing added
- [ ] Cyberduck installed on local Mac
- [ ] Local terminal confirmed working
- [ ] SSH key pair ready

### Notes
- Use one dedicated email if possible for clean project ops.
- Turn on billing alerts in both RunPod and AWS immediately.

---

## A2. Prepare local machine (Mac)

### Install/confirm tools
- [ ] Terminal.app or iTerm2
- [ ] Cyberduck
- [ ] VS Code or Cursor (recommended)
- [ ] Homebrew (optional but useful)
- [ ] Git
- [ ] AWS CLI (later step)

### Generate SSH key
Run:

```bash
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -C "YOUR_EMAIL"
```

Then inspect public key:

```bash
cat ~/.ssh/id_ed25519.pub
```

### Checklist
- [ ] `~/.ssh/id_ed25519` exists
- [ ] `~/.ssh/id_ed25519.pub` exists
- [ ] public key copied

Cyberduck supports public-key auth and can use keys from your SSH setup/agent. citeturn825025search2turn825025search5

---

## A3. Add SSH key to RunPod account

### Actions
- [ ] Open RunPod account settings
- [ ] Find SSH public keys section
- [ ] Paste contents of `id_ed25519.pub`
- [ ] Save

### Why this matters
If the key is in your account **before** Pod creation/startup, RunPod can inject it into the Pod automatically for SSH login. If not, you may need to redeploy or manually add authorized keys. citeturn452509search4

---

## A4. Create the RunPod Pod correctly

### Goal
Deploy a Pod that supports:
- GPU compute
- persistent storage
- direct SSH over exposed TCP
- Jupyter if needed

### Pod selection guidance for Phase 1
Use a modest GPU. You do **not** need a giant machine.

Recommended starting point:
- One official **RunPod PyTorch** template
- A modest GPU with enough VRAM for notebooks and small models
- Keep it simple and cheap first

### Pod creation checklist
- [ ] Choose **Secure Cloud** (preferred for stable/persistent setup)
- [ ] Select an official **PyTorch template** with Jupyter support
- [ ] Confirm **SSH Terminal Access** / direct SSH support is enabled if available
- [ ] Expose **TCP port 22** if needed for SFTP / direct SSH workflows
- [ ] Do **not** rely only on ephemeral container storage
- [ ] Attach persistent storage (next step)

### Why PyTorch template
RunPod’s official PyTorch templates come with JupyterLab preconfigured and support remote IDE workflows. citeturn452509search2turn452509search4

---

## A5. Create and attach persistent storage

### Use a RunPod Network Volume
Why:
- persistent across Pod restarts/redeployments
- good for project code, notebooks, cleaned datasets, outputs
- avoids losing work when the Pod changes

RunPod documents that network volumes are persistent storage and bills them separately; the current published pricing is **$0.07/GB/month for the first 1 TB**. citeturn825025search7

### Recommended starting size
- Start with **100 GB**

That is enough for:
- code
- notebooks
- moderate EOD options data
- plots and checkpoints

### Checklist
- [ ] Create network volume
- [ ] Attach it to Pod at launch if supported
- [ ] Confirm mount path
- [ ] Decide your canonical project root

### Recommended directory layout on remote

```text
/workspace/
  iv_surface_project/
    data_raw/
    data_processed/
    notebooks/
    src/
    reports/
    plots/
    models/
    logs/
```

### Rule
Anything important must live on the persistent volume path, not just somewhere random in the container filesystem.

---

## A6. Verify SSH connectivity

### First test: terminal SSH
From RunPod console, open the Pod’s Connect view and copy the correct SSH command.

Two possibilities exist:

1. **Basic proxied SSH**
   - works for terminal login
   - does **not** support SFTP/SCP

2. **SSH over exposed TCP**
   - needed for Cyberduck SFTP and remote IDE workflows

### Checklist
- [ ] Test basic SSH from terminal
- [ ] Test direct TCP SSH if exposed
- [ ] Confirm you can log in as `root` or the template’s expected user

### If direct TCP is unavailable
You can still start with terminal SSH + `runpodctl` transfer, but since your goal is Cyberduck + a clean workstation, prefer a Pod/template with direct TCP SSH support. RunPod’s docs also recommend `runpodctl` for occasional transfer and SCP/rsync for standard transfers. citeturn452509search3turn452509search4

---

## A7. Configure Cyberduck

### Important caveat
Cyberduck needs **SFTP**. That means you must use the Pod’s **direct TCP SSH endpoint**, not RunPod’s basic proxied SSH endpoint. citeturn452509search0turn452509search5

### Cyberduck bookmark settings
- Protocol: `SFTP (SSH File Transfer Protocol)`
- Server: Pod public IP / hostname from direct TCP section
- Port: exposed SSH port mapped to `:22`
- Username: usually `root` (depends on template)
- Private Key: `~/.ssh/id_ed25519`

### Checklist
- [ ] Create Cyberduck bookmark
- [ ] Test login
- [ ] Save bookmark
- [ ] Confirm you can browse the persistent volume path

### Verification step
Upload a tiny file into the project folder, then reconnect and confirm it is still there.

### Critical check
Confirm you are browsing the persistent directory (for example `/workspace/...`) and not just a temporary folder.

---

## A8. Configure remote editor (recommended)

Cyberduck is good for file transfer, but not your best primary coding interface.

Recommended:
- Cyberduck for drag-and-drop transfers
- VS Code or Cursor Remote-SSH for actual development

RunPod has direct docs for VS Code/Cursor Remote-SSH on Pods that support SSH over exposed TCP. If the Pod stops/resumes, port numbers may change and the SSH config may need updating. citeturn452509search4

### Checklist
- [ ] Install Remote-SSH extension
- [ ] Add Pod host to `~/.ssh/config`
- [ ] Open `/workspace/iv_surface_project`
- [ ] Confirm terminal, file tree, and Python interpreter all work remotely

Suggested `~/.ssh/config` entry:

```ssh
Host <ssh-alias>
  HostName <remote-host>
  User <remote-user>
  Port <remote-port>
  IdentityFile <private-key-path>
```

---

## A9. Initialize project environment on remote

### Create directories

```bash
mkdir -p /workspace/iv_surface_project/{data_raw,data_processed,notebooks,src,reports,plots,models,logs}
```

### Create Python environment
Use `venv` or `conda`. For simplicity, `venv` is fine if the template is already set up.

```bash
cd /workspace/iv_surface_project
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### Install core packages

```bash
pip install numpy scipy matplotlib jupyterlab torch torchvision torchaudio scikit-learn pyarrow fastparquet boto3 awscli
```

Optional:

```bash
pip install pandas seaborn plotly jupyter
```

(If you truly do not want pandas in your workflow, skip it. But practically, options data wrangling often becomes easier with it.)

### Checklist
- [ ] environment created
- [ ] torch import works
- [ ] jupyter works
- [ ] matplotlib works
- [ ] parquet IO works

---

## A10. Add AWS S3 cold backup

### Why S3 is backup, not main workspace
- cheap object storage
- good for archives and snapshots
- bad substitute for an interactive working filesystem

AWS currently publishes **S3 Standard at $0.023/GB/month** for the first 50 TB/month. citeturn825025search3

### Create bucket
- [ ] create one bucket for project backups
- [ ] pick one region and stick to it
- [ ] enable versioning if desired

Suggested bucket name style:
- `ziyuemeng-iv-surface-backups`

### Configure AWS CLI on remote

```bash
aws configure
```

Then test:

```bash
aws s3 ls
```

### Backup commands
Example directory backup:

```bash
aws s3 sync /workspace/iv_surface_project/reports s3://YOUR_BUCKET/reports
aws s3 sync /workspace/iv_surface_project/plots s3://YOUR_BUCKET/plots
aws s3 sync /workspace/iv_surface_project/models s3://YOUR_BUCKET/models
```

### What goes to S3
Put these in S3:
- raw downloaded archives
- processed snapshots
- plots/reports
- model checkpoints you want to keep

Do not rely on S3 as your everyday edit location.

---

## A11. Daily operating workflow

### Start session
- [ ] Start Pod in RunPod console
- [ ] Wait for Pod ready state
- [ ] Check direct SSH port/IP
- [ ] Open remote editor or SSH terminal

### During work
- [ ] Activate environment
- [ ] Pull/update code
- [ ] Run notebook or script
- [ ] Save outputs to `/workspace/iv_surface_project/...`
- [ ] Use Cyberduck only for convenient transfers, not as the sole source of truth

### End of session
- [ ] Sync key outputs/checkpoints to S3
- [ ] Confirm important files are on persistent volume
- [ ] Stop Pod if not actively using GPU

### Cost control rule
Stop compute when you are not actively running jobs. Persistent volume stays; compute charges stop. Network volume persists independently. citeturn825025search7turn452509search2

---

## A12. Remote workstation success criteria

The remote setup is considered done when all of the following are true:

- [ ] You can SSH in reliably
- [ ] You can open the project folder remotely in VS Code/Cursor
- [ ] You can transfer files with Cyberduck over SFTP
- [ ] You can run a Jupyter notebook on the Pod
- [ ] Files persist after stopping/restarting the Pod
- [ ] You can back up outputs to S3

Once these are true, move to Action group B.

---

# 4. Action group B — Lock Phase 1 scope

## B1. Freeze the first underlying
Do **not** start with three names.

Start with:
- [ ] **SPY only**

Why:
- most liquid
- easiest to explain
- lowest project complexity

Only add QQQ/AAPL after the SPY pipeline works.

---

## B2. Freeze the time granularity
Start with:
- [ ] **daily / EOD only**

Do not start with:
- intraday
- minute bars
- full OPRA real-time

---

## B3. Freeze the Phase 1 task
Phase 1 task:

> Given a full real EOD option chain for SPY, construct sparse/noisy/irregular observations and train a model to reconstruct a dense implied-volatility surface or aligned surface representation.

This task definition should remain fixed until the first result package is produced.

---

# 5. Action group C — Build the data pipeline

## C1. Choose data source(s)
Phase 1 data roles:
- **Primary chain data source** = real EOD options chain source
- **Vendor-style reference** = smoothed/traditional reference

### Checklist
- [ ] choose primary chain source
- [ ] choose vendor/reference source
- [ ] document exact schemas available

---

## C2. Decide raw fields needed
At minimum keep:
- [ ] quote date
- [ ] expiration
- [ ] strike
- [ ] call/put flag
- [ ] bid
- [ ] ask
- [ ] mid (compute if needed)
- [ ] underlying spot
- [ ] days-to-maturity or maturity fraction
- [ ] implied vol if provided; otherwise compute later

Optional but useful:
- [ ] volume
- [ ] open interest
- [ ] delta
- [ ] vendor theoretical values

---

## C3. Build ingestion script
Create:
- [ ] one script/notebook to download/load raw chain data
- [ ] save raw snapshots to `data_raw/`
- [ ] write a schema note

Output example:
- raw parquet/csv per date

---

## C4. Clean and normalize chain data
Tasks:
- [ ] remove obviously invalid quotes
- [ ] compute mid if needed
- [ ] remove stale/empty points
- [ ] standardize maturity units
- [ ] standardize moneyness / log-moneyness
- [ ] optionally map to a common surface grid

Deliverable:
- one cleaned dataset for SPY with repeatable preprocessing

---

## C5. Decide target representation
You need to freeze the representation for Phase 1.

Recommended:
- x-axis = moneyness or log-moneyness
- y-axis = maturity
- value = implied volatility

You may use:
1. scattered-point representation first,
2. then map to a standard grid for the neural model.

### Freeze this before modeling
- [ ] coordinate system chosen
- [ ] grid or non-grid design chosen
- [ ] stored format chosen

---

## C6. Produce one clean reference surface per date
Even if the market is not truly “ground truth,” Phase 1 needs a working reference target.

Options:
- cleaned full-chain IV map
- vendor-style smoothed reference
- aligned full-chain proxy surface

### Deliverable
- at least 10–20 dates with clean SPY reference surfaces

---

# 6. Action group D — Define the benchmark task

## D1. Create sparse observation generator
From the full chain/reference on a given day, create partial observations via:
- [ ] random masking
- [ ] structured masking by maturity region
- [ ] structured masking by moneyness region
- [ ] realistic irregular masking patterns

---

## D2. Add quote noise
Create controlled perturbations:
- [ ] low noise regime
- [ ] medium noise regime
- [ ] high noise regime

Goal: make the task reflect real market imperfection.

---

## D3. Save benchmark dataset versions
For reproducibility, save benchmark variants like:
- [ ] `spy_phase1_sparse20_noise0`
- [ ] `spy_phase1_sparse40_noise1`
- [ ] `spy_phase1_sparse60_noise2`

---

## D4. Define evaluation splits
Recommended:
- train dates
- validation dates
- test dates

### Freeze split logic
- [ ] time-based split defined
- [ ] benchmark regimes defined

---

# 7. Action group E — Implement baselines

## E1. Simple non-neural baseline
You need one weak but honest baseline.

Pick one first:
- [ ] linear interpolation
- [ ] nearest-neighbor interpolation
- [ ] simple smoothing

Deliverable:
- one script that takes sparse input and outputs dense surface estimate

---

## E2. Neural baseline
Recommended Phase 1 model:
- [ ] **PyTorch masked neural reconstruction model**

### Suggested first version
A small model with:
- observed values
- mask channel
- coordinate input or grid input
- output dense surface

Reasonable first model choices:
- [ ] small MLP
- [ ] small CNN on gridded surface
- [ ] bottleneck autoencoder / conditional autoencoder

Best narrative choice:
- **conditional autoencoder with mask input**

Why:
- strong DL signal
- easy PyTorch implementation
- natural path to latent-variable Phase 2

---

## E3. Vendor-style reference comparison
You also want comparison against a traditional/industry-style reference.

Tasks:
- [ ] align vendor reference schema with your surface representation
- [ ] define comparison metrics
- [ ] document any mismatch between raw quotes and vendor-smoothed outputs

---

# 8. Action group F — Produce first results

## F1. Metrics
At minimum compute:
- [ ] MAE / RMSE on surface values
- [ ] error on observed points
- [ ] error on unobserved points
- [ ] error by sparsity regime
- [ ] error by maturity bucket
- [ ] error by moneyness bucket

---

## F2. Required plots
Generate at least:
- [ ] reference surface plot
- [ ] sparse/noisy observed plot
- [ ] simple baseline reconstruction plot
- [ ] neural reconstruction plot
- [ ] error vs sparsity plot
- [ ] heatmap of error by region

---

## F3. Required table
One table with rows like:
- simple baseline
- neural baseline
- vendor-style reference gap/alignment

Columns like:
- overall MAE
- MAE observed
- MAE unobserved
- MAE high-sparsity
- MAE short-dated region

---

## F4. Result memo
Write one short internal note answering:
- What worked?
- What failed?
- Where is the task hardest?
- Is simple interpolation enough?
- Why do we need the next phase?

This memo is important. It becomes the bridge to Phase 2.

---

# 9. Definition of “Phase 1 done”

Phase 1 is done when the following are all true:

## Infrastructure
- [ ] remote workstation works end-to-end
- [ ] persistence verified
- [ ] backup flow verified

## Data
- [ ] one real SPY EOD pipeline works
- [ ] one clean processed dataset exists
- [ ] sparse/noisy benchmark generator exists

## Modeling
- [ ] one simple baseline works
- [ ] one PyTorch neural baseline works

## Evaluation
- [ ] metrics computed
- [ ] plots generated
- [ ] one summary table generated

## Narrative
- [ ] one written note summarizing findings exists
- [ ] you can explain what the model was, what the baseline was, and what you compared against

---

# 10. Immediate next actions (ordered)

## Today / first work block
1. [ ] Create/verify RunPod account + billing
2. [ ] Create/verify AWS account + billing alert
3. [ ] Install Cyberduck
4. [ ] Generate SSH key
5. [ ] Add public key to RunPod
6. [ ] Deploy one Secure Cloud PyTorch Pod with direct SSH/TCP support
7. [ ] Create/attach network volume
8. [ ] Verify SSH
9. [ ] Verify Cyberduck SFTP
10. [ ] Create project directories
11. [ ] Create Python env
12. [ ] Install core packages
13. [ ] Create S3 bucket
14. [ ] Test S3 backup

## Next work block
15. [ ] Freeze SPY-only scope
16. [ ] Freeze EOD-only scope
17. [ ] Acquire/load first raw SPY chain dataset
18. [ ] Save raw snapshot
19. [ ] Build cleaning/preprocessing notebook
20. [ ] Produce first clean reference surface

## Next work block after that
21. [ ] Implement sparse masking generator
22. [ ] Implement noise injection
23. [ ] Implement simple interpolation baseline
24. [ ] Implement PyTorch neural baseline
25. [ ] Produce first plots
26. [ ] Write first result memo

---

# 11. What not to do yet

Do **not** do these before the SPY Phase 1 package exists:
- [ ] add too many underlyings
- [ ] do intraday first
- [ ] over-optimize cloud infra
- [ ] build full EBM immediately
- [ ] chase arbitrage-proof guarantees immediately
- [ ] chase production deployment
- [ ] rewrite the project scope every day

---

# 12. Core principle for Phase 1

Phase 1 is not about proving the final theory.

Phase 1 is about producing a **real, runnable, benchmarkable first system** with:
- one real asset,
- one real dataset,
- one neural model,
- one honest comparison,
- one result package.

That is enough to make the project real.
