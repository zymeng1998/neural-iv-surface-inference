# AI Agent Manual — Working on the RunPod Remote Workstation

## Who this document is for

Any AI coding agent (Claude Code, Cursor agent, Copilot, etc.) that needs to execute commands, edit files, or run pipelines on the user's remote RunPod development machine for this project.

Read this entire document before running any remote command.

---

## 1. Connection

### SSH host alias

The RunPod machine is configured in `~/.ssh/config` under the alias:

```
iv-surface-dev-01
```

To run a remote command:

```bash
ssh iv-surface-dev-01 "command here"
```

Do NOT hardcode IPs, ports, usernames, or key paths. Always use the alias. The alias resolves all connection parameters from the user's SSH config.

### Verifying connectivity

Before doing any remote work, run:

```bash
ssh -o ConnectTimeout=10 iv-surface-dev-01 "echo ok"
```

If this fails, stop and tell the user. Common reasons:
- Pod is not running (user needs to start it in RunPod console)
- IP/port changed after a Pod restart (user needs to update `~/.ssh/config`)
- Too many concurrent SSH sessions (wait 2–3 minutes and retry)

### The Pod may not be running

RunPod Pods are stopped between sessions to save cost. If SSH fails with "Connection refused" or "Connection timed out", the Pod is likely stopped. Tell the user to start it from the RunPod console before proceeding.

---

## 2. Post-restart recovery

After every Pod restart, the container filesystem resets but `/workspace` persists. SSH keys and git config need to be restored.

**Always run this first after a restart:**

```bash
ssh iv-surface-dev-01 "/workspace/setup_ssh.sh"
```

This script:
- Copies the GitHub SSH key from `/workspace/.ssh/` to `~/.ssh/`
- Sets correct permissions (`chmod 600`)
- Writes the GitHub SSH config entry

**Then verify GitHub access:**

```bash
ssh iv-surface-dev-01 "ssh -T git@github.com"
```

Expected output: `Hi zymeng1998! You've successfully authenticated...`

**Then restore git identity:**

```bash
ssh iv-surface-dev-01 'git config --global user.name "Ziyue Meng" && git config --global user.email "zymeng1998@gmail.com"'
```

### Quick single-command post-restart setup

```bash
ssh iv-surface-dev-01 '/workspace/setup_ssh.sh && git config --global user.name "Ziyue Meng" && git config --global user.email "zymeng1998@gmail.com" && ssh -T git@github.com 2>&1'
```

---

## 3. Project layout on remote

```
/workspace/                              ← persistent network volume
  Neural-IV-Surface-inference/           ← project git repo (NOTE: capital N, hyphens)
    src/
    data_raw/spy/                        ← downloaded raw data (large, not in git)
    data_processed/spy/                  ← processed pipeline outputs (not in git)
    reports/                             ← generated reports (not in git)
    plots/                               ← generated plots (not in git)
    ...
  setup_ssh.sh                           ← post-restart SSH recovery script
  .ssh/                                  ← persistent SSH keys (not in git)
```

### Critical path detail

The repo directory on remote is:

```
/workspace/Neural-IV-Surface-inference
```

Note the capital `N` and hyphens. This differs from the GitHub repo name (`neural-iv-surface-inference`, lowercase). Always use the actual path, not the repo name.

---

## 4. Running commands on remote

### Simple commands

```bash
ssh iv-surface-dev-01 "command"
```

### Commands with single quotes or complex shell

Use heredoc-style or careful quoting:

```bash
ssh iv-surface-dev-01 'python3 -c "import torch; print(torch.cuda.is_available())"'
```

### Commands that need to run from a specific directory

```bash
ssh iv-surface-dev-01 "cd /workspace/Neural-IV-Surface-inference && command"
```

### Running Python scripts

```bash
ssh iv-surface-dev-01 "cd /workspace/Neural-IV-Surface-inference/src/data && python -u script.py"
```

Always use `python -u` (unbuffered) for long-running scripts so output streams correctly.

---

## 5. Avoiding common pitfalls

### Pitfall 1: Long-running SSH commands get backgrounded or timeout

SSH commands that take more than ~60 seconds may be automatically backgrounded by the agent harness, making output invisible.

**Workaround**: For long-running commands, redirect output to a file on the remote, then read it:

```bash
# Start the job
ssh iv-surface-dev-01 "cd /workspace/Neural-IV-Surface-inference/src/data && python -u script.py > /workspace/job_output.log 2>&1 && echo DONE >> /workspace/job_output.log || echo FAILED >> /workspace/job_output.log"

# Check status later
ssh iv-surface-dev-01 "tail -20 /workspace/job_output.log"
```

Or use `nohup` so the job survives SSH disconnection:

```bash
ssh iv-surface-dev-01 "cd /workspace/Neural-IV-Surface-inference/src/data && nohup python -u script.py > /workspace/job_output.log 2>&1 &"

# Check if still running
ssh iv-surface-dev-01 "ps aux | grep script.py | grep -v grep"

# Read output
ssh iv-surface-dev-01 "cat /workspace/job_output.log"
```

### Pitfall 2: Too many concurrent SSH sessions crash the Pod

The Pod has limited resources. Opening many parallel SSH connections (especially while running memory-heavy Python jobs) can cause OOM or SSH daemon saturation.

**Rules:**
- Never run more than 2 concurrent SSH commands
- Never start a new heavy job while another is still running
- If SSH starts timing out after previously working, wait 2–3 minutes before retrying
- If the Pod becomes fully unresponsive, tell the user to restart it from the RunPod console

### Pitfall 3: Memory limits

The Pod has a single GPU (RTX A5000, 24 GB VRAM) and limited system RAM. Loading the full SPY options dataset (~24.7M rows, ~632 MB Parquet, expands to several GB in pandas) can push memory limits.

**Rules:**
- Do not load the full dataset multiple times in parallel
- Use `del df; import gc; gc.collect()` after heavy operations
- If a script crashes silently, check for OOM: `ssh iv-surface-dev-01 "dmesg | tail -20"`

### Pitfall 4: Network volume permissions

The `/workspace` network volume uses a filesystem that does NOT support `chmod`. All files on it have permissions `0666` (rw-rw-rw-).

**This means:**
- SSH private keys cannot be stored directly on `/workspace` with correct permissions
- Keys must be COPIED (not symlinked) from `/workspace/.ssh/` to `~/.ssh/` (which is on the container filesystem and supports chmod)
- The `setup_ssh.sh` script handles this
- Never symlink keys from `/workspace` — SSH will reject them with "UNPROTECTED PRIVATE KEY FILE"

### Pitfall 5: Container filesystem resets on restart

Everything outside `/workspace` is ephemeral:
- `~/.ssh/config` — gone after restart
- `~/.gitconfig` — gone after restart
- Installed pip packages — may or may not survive depending on template
- System configs — gone after restart

**Rule**: Anything that needs to survive restarts must live under `/workspace`.

---

## 6. Git workflow — three-way sync

There are three copies of the repo:

| Location | Path | Role |
|----------|------|------|
| User's Mac (local) | `~/Neural IV Surface inference/` | Local editing, commits |
| GitHub | `zymeng1998/neural-iv-surface-inference` | Central remote |
| RunPod | `/workspace/Neural-IV-Surface-inference/` | Remote execution |

### Sync protocol

After making changes on any side:

1. **Commit + push** from the changed location
2. **Pull** on the other locations

```bash
# If you changed files locally → push, then pull on RunPod:
cd "~/Neural IV Surface inference" && git push origin main
ssh iv-surface-dev-01 "cd /workspace/Neural-IV-Surface-inference && git pull --ff-only"

# If you changed files on RunPod → push, then pull locally:
ssh iv-surface-dev-01 "cd /workspace/Neural-IV-Surface-inference && git push origin main"
cd "~/Neural IV Surface inference" && git pull --ff-only
```

### Verifying sync

```bash
# Check all three HEADs match:
LOCAL=$(cd "~/Neural IV Surface inference" && git rev-parse --short HEAD)
GITHUB=$(cd "~/Neural IV Surface inference" && git ls-remote origin HEAD | cut -c1-7)
RUNPOD=$(ssh iv-surface-dev-01 "cd /workspace/Neural-IV-Surface-inference && git rev-parse --short HEAD")
echo "Local:  $LOCAL"
echo "GitHub: $GITHUB"
echo "RunPod: $RUNPOD"
```

### Important: check for uncommitted work before pulling

```bash
ssh iv-surface-dev-01 "cd /workspace/Neural-IV-Surface-inference && git status --short"
```

If there are uncommitted changes on RunPod, either commit them first or stash them.

---

## 7. Python environment on remote

The Pod uses the system Python from the RunPod PyTorch template. There is no virtualenv.

- Python: 3.11.10
- PyTorch: 2.4.1+cu124
- CUDA: 12.4
- GPU: NVIDIA RTX A5000 (24 GB VRAM)

### Installed packages (as of 2026-03-31)

numpy, scipy, matplotlib, scikit-learn, pandas, pyarrow, fastparquet, boto3, awscli, seaborn, yfinance, jupyterlab, torch, torchvision, torchaudio

### Installing new packages

```bash
ssh iv-surface-dev-01 "pip install package_name"
```

Note: pip packages installed via the system pip may not survive Pod restarts depending on the template. If a package is critical, add it to a requirements file and re-install after restart.

---

## 8. File transfer

### Small files: use SSH

```bash
# Local → remote
scp -P <remote-port> local_file.py <remote-user>@<remote-host>:<workspace-path>/Neural-IV-Surface-inference/

# remote → Local
scp -P <remote-port> <remote-user>@<remote-host>:<workspace-path>/file.txt ./
```

**Note**: SCP requires the direct TCP SSH endpoint (host + port from the provider console). These values are intentionally not committed — keep them in a local-only private runbook. The port may change after restart. Prefer git for code and a GUI SFTP client for data files.

### Code: use git

Commit and push/pull. This is the preferred method for all code changes.

### Large files / data: use Cyberduck

The user has Cyberduck configured for SFTP to the Pod. Tell the user to use Cyberduck for large file transfers rather than trying to SCP through an agent.

---

## 9. Security rules

**Never include in any tracked/committed file:**
- Pod IP address
- SSH port number
- SSH username
- Full SSH config blocks
- Private key file paths
- API keys, tokens, passwords
- Any literal connection string

**Safe to reference in tracked files:**
- SSH alias name (`iv-surface-dev-01`)
- Provider name (RunPod)
- Workspace path (`/workspace`)
- Project repo path (`/workspace/Neural-IV-Surface-inference`)
- Tool names (Cyberduck, Cursor Remote-SSH)
- Package versions
- Non-sensitive workflow descriptions

---

## 10. Troubleshooting checklist

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| SSH "Connection refused" | Pod not running | User starts Pod in RunPod console |
| SSH "Connection timed out" | Pod overloaded or IP changed | Wait 3 min and retry, or user checks RunPod console for new IP/port |
| SSH "Permission denied" | Wrong key or key not in RunPod account | Use alias `iv-surface-dev-01`; if still fails, user checks RunPod SSH key settings |
| `git push` fails on RunPod | GitHub SSH not configured post-restart | Run `/workspace/setup_ssh.sh` |
| Python import fails | Package not installed or lost after restart | `pip install package_name` |
| Script crashes silently | OOM | Check `dmesg | tail`, reduce data size, restart Pod |
| "UNPROTECTED PRIVATE KEY FILE" | Key symlinked from /workspace | Copy instead of symlink; run `/workspace/setup_ssh.sh` |
| Parquet files not found | Pipeline steps not run in order | Run `01_ingest` → `02_inspect` → `03_build` sequentially |

---

## 11. Quick reference — copy-paste recipes

### Full post-restart recovery

```bash
ssh iv-surface-dev-01 '/workspace/setup_ssh.sh && git config --global user.name "Ziyue Meng" && git config --global user.email "zymeng1998@gmail.com" && echo "--- GitHub auth ---" && ssh -T git@github.com 2>&1 && echo "--- GPU check ---" && python3 -c "import torch; print(f\"CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}\")"'
```

### Three-way sync check

```bash
echo "LOCAL:  $(cd "$HOME/Neural IV Surface inference" && git rev-parse --short HEAD)" && echo "GITHUB: $(cd "$HOME/Neural IV Surface inference" && git ls-remote origin HEAD | cut -c1-7)" && echo "RUNPOD: $(ssh iv-surface-dev-01 "cd /workspace/Neural-IV-Surface-inference && git rev-parse --short HEAD" 2>/dev/null)"
```

### Run a Python script safely on remote

```bash
ssh iv-surface-dev-01 "cd /workspace/Neural-IV-Surface-inference && python -u path/to/script.py 2>&1 | tee /workspace/last_run.log"
```

### Check what's running on remote

```bash
ssh iv-surface-dev-01 "ps aux --sort=-%mem | head -15"
```
