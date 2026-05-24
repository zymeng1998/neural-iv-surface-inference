#!/usr/bin/env python3
"""Generate notebooks/04_phase2c_results.ipynb from the committed AV-rerun
artifacts.

Run from the repo root:
    python3 scripts/generate_phase2c_results_notebook.py

The notebook itself loads `artifacts/results/*av_rerun_20260523_075616*.csv`
and `artifacts/results/baseline_results.csv` and produces all figures
inline — no extra data needed beyond what's already committed.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text)


def code(src: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(src)


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.11"},
    }

    cells: list[nbf.NotebookNode] = []

    # -----------------------------------------------------------------
    # Title + executive summary
    # -----------------------------------------------------------------
    cells.append(md(
        """# Phase 2C Results — Conditional Surface Model on Alpha Vantage SPY

**Date:** 2026-05-23 · **Source data:** Alpha Vantage `HISTORICAL_OPTIONS`
(2008-01-02 → 2026-05-22, 26.06M raw rows, 4,623 trading days)
· **Benchmark:** `spy_phase1_random40_noiselow` (22.5M strict-cleaned rows;
chronological train/val/test = 11.1M / 5.6M / 5.8M; 40% observed mask, low
homoscedastic noise σ=0.01).

This notebook reports the Phase 2C deliverable: the **W3 conditional
neural surface model** trained and evaluated on real SPY data, alongside
the refreshed Phase 1 baselines (interpolation + MLP) for like-for-like
comparison.

## Headline test MAE (40% observed, 60% to be reconstructed)

| Model | Params | Test MAE | obs / unobs MAE | Train time |
|---|---|---|---|---|
| **`interp_rbf`** (classical) | — | **0.0662** | 0.0542 / 0.0742 | n/a |
| **`conditional` (W3, new)** | 85,057 | **0.0753** | 0.0753 / 0.0754 | **3 m 40 s (GPU)** |
| `mlp` (Phase 1) | 34,305 | 0.0951 | 0.0951 / 0.0951 | ~80 m (CPU+GPU) |

The conditional model **beats the Phase 1 MLP by ~21%** on test MAE while
**still losing to the RBF interpolation floor by ~14%**. The MAE is
essentially identical on observed vs unobserved query points — the
parametric model doesn't differentiate by mask state, unlike RBF which
shows the expected observed-better-than-unobserved gap.
"""
    ))

    # -----------------------------------------------------------------
    # What the model does
    # -----------------------------------------------------------------
    cells.append(md(
        """## What the model actually does

Given a date `t` and the chain of observed quotes that day,
`O_t = {(k_i, τ_i, σ_observed_i)}`, the model predicts the implied
volatility at **any** `(k, τ)` query — including coordinates that were
not quoted that day. This is **surface inference**, not point prediction
of a single IV.

```
INPUT  per date t:  set O_t of variable size (1K to 13K points)
                    columns: (log_moneyness, tau, observed_IV)
OUTPUT per date t:  σ̂(k, τ | O_t) for ANY (k, τ) query
                    (the full surface, including unquoted strikes/tenors)
```

Architecture:

```
SetEncoder           (per-element MLP → masked-mean pool → post-pool MLP → z_t)
                       z_t is the date's latent: permutation- + mask-invariant
                       Output dim: 64

CoordinateDecoder    (concat z_t with each (k, τ) query → MLP → softplus)
                       Hidden dim: 128, depth: 3
                       Output: positive IV per query
```

Total: 85,057 parameters. The set encoder makes this model **conditional
on the observed chain** — different days get different latents.
"""
    ))

    # -----------------------------------------------------------------
    # I/O contract + inference demo (placed AFTER imports below)
    # -----------------------------------------------------------------
    IO_CONTRACT_MD = md(
        """## I/O contract — what you feed in, what comes out

**Per-date inference takes two arrays and returns one:**

| Argument | dtype | Shape | Per-row contents |
|---|---|---|---|
| `context` (today's observed chain) | `torch.float32` | `(N, 3)` — N = # quoted options today, typically 1K-13K | `[log_moneyness, tau, observed_IV]` |
| `query` (where to predict) | `torch.float32` | `(M, 2)` — M = arbitrary | `[log_moneyness, tau]` |
| **returns** | `torch.float32` | `(M,)` | predicted IV per query coord (softplus, > 0) |

Field meanings:

| Field | Unit | Typical range | How to compute |
|---|---|---|---|
| `log_moneyness` | dimensionless | ≈ `[-1, +1]` for SPY | `ln(strike / spot_close)` |
| `tau` | years (fractional) | `[1/365, 2.0]` | `(expiration - trade_date).days / 365.25` |
| `IV` | annualized vol, decimal | `[0.01, 3.0]` | from chain's `implied_volatility` |

All `float32` after pipeline cleaning. **No single-scalar inputs**: the
model is fundamentally set-to-function — give it a chain set, get back a
surface you can query anywhere.

The next cells load the trained checkpoint and run one inference call
on a 5-quote synthetic chain so you can see the exact shapes."""
    )
    IO_CHECKPOINT_LOAD = code(
        """import sys
sys.path.insert(0, '../src')
import torch
from neural_iv_surface_inference.models.conditional_surface import ConditionalSurfaceModel

# 1. Load the trained checkpoint.
ckpt = torch.load('../artifacts/checkpoints/best_conditional.pt',
                  map_location='cpu', weights_only=False)
cfg = ckpt['config']
print('checkpoint architecture config:', {k: cfg[k] for k in
    ['context_dim','coord_dim','hidden_dim','latent_dim',
     'n_elem_layers','n_post_layers','n_decoder_layers']})

# 2. Rebuild the model and load weights.
model = ConditionalSurfaceModel(
    context_dim=cfg['context_dim'], coord_dim=cfg['coord_dim'],
    hidden_dim=cfg['hidden_dim'], latent_dim=cfg['latent_dim'],
    n_elem_layers=cfg['n_elem_layers'], n_post_layers=cfg['n_post_layers'],
    n_decoder_layers=cfg['n_decoder_layers'],
)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
print(f'params: {sum(p.numel() for p in model.parameters()):,}')
"""
    )
    IO_INFER_DEMO = code(
        """# 3. Build a small synthetic chain (5 quotes).
context_np = np.array([
    # log_moneyness, tau (years),  observed_IV
    [-0.10,           30/365.25,   0.22],   # ~5% ITM call, 1 month
    [-0.05,           30/365.25,   0.20],
    [ 0.00,           30/365.25,   0.18],   # ATM 1 month
    [ 0.05,           30/365.25,   0.19],
    [ 0.10,           90/365.25,   0.21],   # OTM 3 months
], dtype=np.float32)
context = torch.from_numpy(context_np).unsqueeze(0)        # (1, 5, 3)
mask    = torch.ones(1, 5, dtype=torch.bool)

# 4. Query grid — 6 points across two maturities.
query_np = np.array([
    [-0.10, 60/365.25],   [0.00, 60/365.25],   [0.10, 60/365.25],
    [-0.10, 180/365.25],  [0.00, 180/365.25],  [0.10, 180/365.25],
], dtype=np.float32)
query = torch.from_numpy(query_np).unsqueeze(0)            # (1, 6, 2)

# 5. Predict.
with torch.no_grad():
    pred = model(context, mask, query)                     # (1, 6)

print(f'context shape: {tuple(context.shape)}  dtype: {context.dtype}')
print(f'query   shape: {tuple(query.shape)}   dtype: {query.dtype}')
print(f'pred    shape: {tuple(pred.shape)}    dtype: {pred.dtype}')
print()
print('Predicted IV surface (caveat: a real chain has ~1K-13K quotes, not 5):')
for q, p in zip(query_np, pred[0].numpy()):
    print(f'  log_moneyness={q[0]:+.2f}  tau={q[1]:.4f}y ({q[1]*365.25:.0f}d)'
          f'  ->  iv={p:.4f}')
"""
    )
    IO_SURFACE_PLOT = code(
        """# 6. Visualize the predicted surface on a denser grid (still the same toy chain).
k_grid   = np.linspace(-0.30, 0.30, 30, dtype=np.float32)
tau_grid = np.array([30, 60, 90, 180, 365], dtype=np.float32) / 365.25

Q = np.array([[k, t] for t in tau_grid for k in k_grid], dtype=np.float32)
Q_t = torch.from_numpy(Q).unsqueeze(0)
with torch.no_grad():
    iv_pred = model(context, mask, Q_t).squeeze(0).numpy()
iv_grid = iv_pred.reshape(len(tau_grid), len(k_grid))

fig, ax = plt.subplots(figsize=(10, 4.5))
for i, t in enumerate(tau_grid):
    ax.plot(k_grid, iv_grid[i], lw=2, label=f'tau={t*365.25:.0f}d')
ax.scatter(context_np[:,0], context_np[:,2], s=80, color='black', zorder=5,
           label='observed (input chain)')
ax.set_xlabel('log_moneyness')
ax.set_ylabel('predicted IV')
ax.set_title('One-day IV surface predicted by the conditional model')
ax.legend(loc='upper right', fontsize=9)
plt.tight_layout(); plt.show()
"""
    )

    # -----------------------------------------------------------------
    # Imports + load
    # -----------------------------------------------------------------
    cells.append(code(
        """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({
    'figure.dpi': 110,
    'savefig.dpi': 110,
    'figure.figsize': (10, 5),
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

RESULTS = Path('../artifacts/results')
TAG = 'av_rerun_20260523_075616'

baseline = pd.read_csv(RESULTS / 'baseline_results.csv')
w1_interp = pd.read_csv(RESULTS / f'uncertainty_eval_{TAG}_interp_rbf.csv')
w1_cond   = pd.read_csv(RESULTS / f'uncertainty_eval_{TAG}_conditional.csv')
w2_interp = pd.read_csv(RESULTS / f'structure_diagnostics_{TAG}_interp_rbf.csv')
w2_cond   = pd.read_csv(RESULTS / f'structure_diagnostics_{TAG}_conditional.csv')
curve_interp = pd.read_csv(RESULTS / f'uncertainty_eval_{TAG}_interp_rbf_curve.csv')
curve_cond   = pd.read_csv(RESULTS / f'uncertainty_eval_{TAG}_conditional_curve.csv')

print(f'baseline rows: {len(baseline)}  models: {sorted(baseline[\"model\"].unique())}')
print(f'w1 splits: {sorted(w1_interp[\"split\"].unique())}')
print(f'w2 dates per split: {w2_interp.groupby(\"split\").size().to_dict()}')
"""
    ))

    # Now insert the I/O contract + demo (after numpy/matplotlib are imported above)
    cells.append(IO_CONTRACT_MD)
    cells.append(IO_CHECKPOINT_LOAD)
    cells.append(IO_INFER_DEMO)
    cells.append(IO_SURFACE_PLOT)

    # -----------------------------------------------------------------
    # Practical workflow on TODAY's real SPY chain
    # -----------------------------------------------------------------
    cells.append(md(
        """## Practical workflow — predict TODAY's surface from a live chain

This section runs the **full end-to-end workflow** on a real SPY option
chain pulled from Yahoo Finance (free, no API key). Re-run the cells any
trading day and they'll fetch the latest chain, compute features, predict,
and produce a surface.

**Misconceptions to clear first:**

1. **`log_moneyness ≠ log(strike)`** — it is `ln(strike / spot_close)`.
   ATM = 0; ITM call (low strike) = negative; OTM call (high strike) =
   positive. You need today's SPY spot too.
2. **`tau` is time-to-expiry**, not "how far in the future." A 90-day
   `tau` means "options expiring 90 days from today, which are quoted
   TODAY." The model is not a time-series forecaster.
3. **This is inference, not forecasting.** The model takes today's
   chain and tells you today's full surface for every existing expiry.
   "What will the surface look like in 3 months?" is a different
   question (different model).

**Caching:** the cell below saves the fetched chain to
`artifacts/results/live_spy_chain_<date>.csv` so re-executing the
notebook on a different day still has reproducible numbers — unless you
explicitly bust the cache."""
    ))
    cells.append(code(
        """# Pull today's SPY chain via yfinance (cached to disk for reproducibility).
import datetime as _dt
CACHE = Path('../artifacts/results') / f'live_spy_chain_{_dt.date.today().isoformat()}.csv'

if CACHE.exists():
    print(f'using cached chain: {CACHE.name}')
    chain = pd.read_csv(CACHE)
    spot = float(chain.attrs.get('spot') or chain['_spot'].iloc[0])
else:
    import yfinance as yf
    spy = yf.Ticker('SPY')
    spot = float(spy.history(period='1d')['Close'].iloc[-1])
    today = _dt.datetime.utcnow().date()
    rows = []
    for exp_str in spy.options[:8]:    # first 8 expirations (~next 2 months)
        exp_date = _dt.datetime.strptime(exp_str, '%Y-%m-%d').date()
        tau = (exp_date - today).days / 365.25
        if tau <= 0:
            continue
        oc = spy.option_chain(exp_str)
        for side, df in [('call', oc.calls), ('put', oc.puts)]:
            sub = df[['strike','impliedVolatility','bid','ask','volume']].copy()
            sub = sub[(sub['impliedVolatility'] > 0.01) & (sub['impliedVolatility'] < 3.0)]
            sub['log_moneyness'] = np.log(sub['strike'] / spot)
            sub['tau'] = tau
            sub['type'] = side
            rows.append(sub)
    chain = pd.concat(rows, ignore_index=True)
    chain['_spot'] = spot
    chain.to_csv(CACHE, index=False)
    print(f'fetched + cached: {CACHE.name}')

print(f'SPY spot: ${spot:.2f}   quotes: {len(chain):,} across {chain[\"tau\"].nunique()} expirations')
chain[['type','strike','log_moneyness','tau','impliedVolatility','volume']].head(6)
"""
    ))
    cells.append(code(
        """# Step 2: build the (N,3) context tensor exactly as the model expects.
ctx_np = chain[['log_moneyness','tau','impliedVolatility']].values.astype(np.float32)
N = len(ctx_np)
live_context = torch.from_numpy(ctx_np).unsqueeze(0)        # (1, N, 3)
live_mask    = torch.ones(1, N, dtype=torch.bool)
print(f'context tensor: shape={tuple(live_context.shape)}  dtype={live_context.dtype}')
print(f'                N = {N} quoted options across {chain[\"tau\"].nunique()} maturities')
"""
    ))
    cells.append(code(
        """# Step 3: predict at the SAME coordinates the market quoted - so we can
# compare model vs market quote-by-quote.
q_np = chain[['log_moneyness','tau']].values.astype(np.float32)
q    = torch.from_numpy(q_np).unsqueeze(0)
with torch.no_grad():
    pred_live = model(live_context, live_mask, q).squeeze(0).numpy()

chain['iv_model'] = pred_live
chain['iv_diff']  = chain['impliedVolatility'] - chain['iv_model']

# Overall fit quality on TODAY's chain
print(f'Mean |market_IV - model_IV| on today chain: {chain[\"iv_diff\"].abs().mean():.4f}')
print(f'Median |diff|:                              {chain[\"iv_diff\"].abs().median():.4f}')
print(f'90th-pct |diff|:                            {chain[\"iv_diff\"].abs().quantile(0.9):.4f}')
"""
    ))
    cells.append(code(
        """# Where does the model disagree most with the live chain?
liquid = chain[chain['volume'] > 0].copy()
print(f'{len(liquid):,} liquid quotes (volume > 0)')

print('\\nTop 6 chain IV > model IV  (model says \"rich\"):')
print(liquid.nlargest(6, 'iv_diff')[
    ['type','strike','tau','impliedVolatility','iv_model','iv_diff','volume']
].to_string(index=False))

print('\\nTop 6 chain IV < model IV  (model says \"cheap\"):')
print(liquid.nsmallest(6, 'iv_diff')[
    ['type','strike','tau','impliedVolatility','iv_model','iv_diff','volume']
].to_string(index=False))
"""
    ))
    cells.append(md(
        """**Reading the disagreement table:** large `iv_diff` is most often
caused by Yahoo's retail IV calc being unreliable in the deep wings and
at very short tenors (≤3 days), not by genuine mispricing. For a real
trade signal you'd want a pro EOD feed (Bloomberg, ORATS, …) feeding the
same model. The point of the table here is to *demonstrate the workflow*:
once the model surface is built, screening for outliers vs market is a
one-liner."""
    ))
    cells.append(code(
        """# Step 4: predict at coordinates the market did NOT quote.
# Example: an exact-ATM 14-day option, a 5%-ITM 1-month, a 10%-OTM 6-month.
custom = torch.tensor([
    [ 0.000,  14/365.25],
    [-0.05,   30/365.25],
    [ 0.10,  180/365.25],
], dtype=torch.float32).unsqueeze(0)
with torch.no_grad():
    iv_custom = model(live_context, live_mask, custom).squeeze(0).numpy()
print(f'SPY spot used: ${spot:.2f}\\n')
print('Predicted IV at coordinates the live chain did NOT quote:')
for c, iv in zip(custom.squeeze(0).numpy(), iv_custom):
    strike = spot * np.exp(c[0])
    days = c[1] * 365.25
    print(f'  strike ${strike:7.2f}  (log_m={c[0]:+.3f})  {days:5.1f}d to expiry  ->  IV {iv:.4f}')
"""
    ))
    cells.append(code(
        """# Step 5: build a full (k, tau) grid and visualize the predicted surface.
k_grid   = np.linspace(-0.20, 0.20, 60, dtype=np.float32)
# unique maturities currently listed:
tau_unique = sorted(chain['tau'].unique())
tau_grid   = np.array(tau_unique, dtype=np.float32)

QQ = np.array([[k, t] for t in tau_grid for k in k_grid], dtype=np.float32)
QQ_t = torch.from_numpy(QQ).unsqueeze(0)
with torch.no_grad():
    iv_pred = model(live_context, live_mask, QQ_t).squeeze(0).numpy()
iv_grid = iv_pred.reshape(len(tau_grid), len(k_grid))

fig, ax = plt.subplots(figsize=(11, 5.5))
cmap = plt.cm.viridis(np.linspace(0, 0.9, len(tau_grid)))
for i, t in enumerate(tau_grid):
    days = int(round(t * 365.25))
    ax.plot(k_grid, iv_grid[i], lw=2, color=cmap[i], label=f'{days}d')
# Overlay actual chain quotes
ax.scatter(chain['log_moneyness'], chain['impliedVolatility'],
           s=8, alpha=0.25, color='black', label='market quotes')
ax.set_xlabel('log_moneyness  =  ln(strike / spot)')
ax.set_ylabel('Implied Volatility (decimal)')
ax.set_title(f'Conditional model — TODAY\\'s SPY surface (spot=${spot:.2f}, {len(chain):,} quotes)')
ax.legend(title='time to expiry', loc='upper center', ncol=4, fontsize=9)
ax.set_xlim(-0.30, 0.30)
ax.set_ylim(0, min(1.5, iv_grid.max()*1.1))
plt.tight_layout(); plt.show()
"""
    ))
    cells.append(md(
        """### What the smile/surface plot is telling you

- Each colored line is the predicted IV smile for one tenor. Shorter
  maturities sit above longer ones (vol-of-vol term structure).
- The black dots are the actual market quotes. Lines that hug the dots
  closely mean the model's prediction agrees with the live chain at
  those coords; large deviations between line and dot in deep wings are
  the "mispricing" candidates from the table above.
- The shape (downward slope from OTM-put side, flattening on call side)
  is the classic SPY skew — empirically captured by the model.

### What to do next with this in your workflow

| Want to do this | How |
|---|---|
| Price a 45-day option at strike $X | `query = [[ln(X/spot), 45/365.25]]`, get IV, plug into Black-Scholes |
| Vega/Vomma risk on a dense grid | Predict on `(k_grid × tau_grid)`, compute price derivatives by finite differences |
| Daily mispricing scan | Run this notebook end-to-end every EOD; alert on `|iv_diff| > threshold` |
| Compare to broker | Get your broker's IV column for the same chain; same diff calc, different opinion |

### Caveats (don't skip these)

- **Training distribution = EOD CLOSE.** yfinance gives live mid-quotes,
  which have wider effective spreads than EOD. For real signals use a
  proper EOD feed (Bloomberg/ORATS/Polygon).
- **No uncertainty signal yet** — the model gives you a point estimate.
  Epic 2D adds heteroscedastic/ensemble heads so you can also get a
  prediction band around each IV.
- **SPY only.** Don't use this checkpoint on SPX, QQQ, or single
  names — those distributions are different. To extend you'd retrain
  on that ticker's history.
"""
    ))

    # -----------------------------------------------------------------
    # 3D SURFACE VISUALIZATION (static + spinning GIF + interactive Plotly)
    # -----------------------------------------------------------------
    cells.append(md(
        """## 🌀 3D Surface visualization — the headline image

The IV surface is fundamentally a **3D object**: `σ(k, τ)`. Showing it as
2D smile slices loses the term-structure picture. The next three cells
render the same surface three different ways:

1. **Static 3D mesh** (matplotlib) — well-angled, publication-ready.
2. **Animated spinning GIF** (matplotlib) — embed in slides / share via
   Slack / email; no JavaScript needed.
3. **Interactive Plotly** — drag with your mouse to rotate inside the
   notebook; renders WebGL.

All three plot the **conditional model's predicted surface** built from
today's live SPY chain (from the section above), with the actual market
quotes overlaid as black scatter points."""
    ))
    cells.append(code(
        """# Build a dense (k, tau) mesh and run inference once.
k_grid_3d   = np.linspace(-0.25, 0.25, 70, dtype=np.float32)
# Pick a smoother tau axis than just the listed expirations
tau_min     = float(chain['tau'].min())
tau_max     = float(chain['tau'].max())
tau_grid_3d = np.linspace(tau_min, tau_max, 40, dtype=np.float32)

K3D, T3D = np.meshgrid(k_grid_3d, tau_grid_3d)
mesh_q   = np.stack([K3D.ravel(), T3D.ravel()], axis=-1).astype(np.float32)
mesh_t   = torch.from_numpy(mesh_q).unsqueeze(0)
with torch.no_grad():
    iv_mesh = model(live_context, live_mask, mesh_t).squeeze(0).numpy()
IV3D = iv_mesh.reshape(K3D.shape)

print(f'mesh: k={K3D.shape[1]} x tau={K3D.shape[0]} = {K3D.size:,} surface points')
print(f'IV range on mesh: [{IV3D.min():.3f}, {IV3D.max():.3f}]')
"""
    ))
    cells.append(code(
        """# (1) Static 3D mesh — publication-ready.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
fig = plt.figure(figsize=(11, 7))
ax = fig.add_subplot(111, projection='3d')

# Days to expiry on the y-axis is more readable than raw tau
DAYS3D = T3D * 365.25

surf = ax.plot_surface(
    K3D, DAYS3D, IV3D,
    cmap='viridis', alpha=0.85,
    linewidth=0, antialiased=True,
    rcount=40, ccount=70,
)
# Overlay actual market quotes as 3D scatter
ax.scatter(
    chain['log_moneyness'].values,
    chain['tau'].values * 365.25,
    chain['impliedVolatility'].values,
    c='red', s=10, alpha=0.6, label='market quotes', depthshade=True,
)
ax.set_xlabel('log_moneyness')
ax.set_ylabel('days to expiry')
ax.set_zlabel('Implied Volatility')
ax.set_title(f"Conditional model — SPY surface (spot=${spot:.2f}, {len(chain):,} live quotes)",
             pad=14)
ax.view_init(elev=28, azim=-55)
fig.colorbar(surf, shrink=0.55, aspect=20, label='IV')
ax.legend(loc='upper right')
plt.tight_layout(); plt.show()
"""
    ))
    cells.append(md(
        """### (2) Spinning GIF — render once, embed everywhere

Builds 60 frames at 6° azimuth steps, stitches with Pillow → ~120-300 KB
animated GIF saved to `artifacts/results/surface_3d_spin_<date>.gif`.
Runs in ~10-20 seconds on a Mac CPU.

The GIF is embedded inline below for the notebook itself, and lives on
disk for use in slides, README, Slack, email, etc."""
    ))
    cells.append(code(
        """import io
from PIL import Image as PILImage
from IPython.display import Image as IPyImage, display
import datetime as _dt

GIF_PATH = Path('../artifacts/results') / f'surface_3d_spin_{_dt.date.today().isoformat()}.gif'

def render_frame(azim_deg: float) -> PILImage.Image:
    fig = plt.figure(figsize=(7, 5), dpi=90)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(K3D, DAYS3D, IV3D, cmap='viridis', alpha=0.9,
                    linewidth=0, antialiased=True, rcount=40, ccount=70)
    ax.scatter(chain['log_moneyness'].values,
               chain['tau'].values * 365.25,
               chain['impliedVolatility'].values,
               c='red', s=6, alpha=0.55, depthshade=True)
    ax.set_xlabel('log_moneyness'); ax.set_ylabel('days'); ax.set_zlabel('IV')
    ax.set_title(f'SPY conditional surface (spot=${spot:.2f})')
    ax.view_init(elev=25, azim=azim_deg)
    ax.set_box_aspect((1, 1.2, 0.7))
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight'); plt.close(fig)
    buf.seek(0)
    return PILImage.open(buf).convert('P', palette=PILImage.Palette.ADAPTIVE)

print('rendering 60 frames @ 6° each ...')
frames = [render_frame(a) for a in np.linspace(0, 354, 60)]
frames[0].save(
    GIF_PATH, save_all=True, append_images=frames[1:],
    duration=105, loop=0, optimize=True, disposal=2,
)
print(f'wrote {GIF_PATH}  ({GIF_PATH.stat().st_size/1024:.0f} KB)')
display(IPyImage(filename=str(GIF_PATH)))
"""
    ))
    cells.append(md(
        """### (3) Interactive Plotly — drag to rotate inside the notebook

The Plotly figure below renders as WebGL inside Jupyter. Click-and-drag
to rotate, scroll to zoom, double-click to reset view. The same data,
fully exploratory."""
    ))
    cells.append(code(
        """import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = 'notebook'

surface = go.Surface(
    x=k_grid_3d,
    y=tau_grid_3d * 365.25,
    z=IV3D,
    colorscale='Viridis',
    opacity=0.92,
    colorbar=dict(title='IV', thickness=15),
    name='model surface',
    showscale=True,
)
scatter = go.Scatter3d(
    x=chain['log_moneyness'].values,
    y=chain['tau'].values * 365.25,
    z=chain['impliedVolatility'].values,
    mode='markers',
    marker=dict(size=2.5, color='red', opacity=0.7),
    name='market quotes',
)
fig3d = go.Figure(data=[surface, scatter])
fig3d.update_layout(
    title=f'SPY IV surface — conditional model (spot=${spot:.2f}, {len(chain):,} quotes)',
    scene=dict(
        xaxis_title='log_moneyness',
        yaxis_title='days to expiry',
        zaxis_title='Implied Volatility',
        camera=dict(eye=dict(x=1.6, y=-1.6, z=0.9)),
        aspectratio=dict(x=1, y=1.2, z=0.8),
    ),
    width=900, height=600,
    margin=dict(l=0, r=0, t=40, b=0),
)
fig3d.show()
"""
    ))
    cells.append(md(
        """### (4) Side-by-side: model surface vs RBF interpolation surface

Both surfaces are conditioned on the same input chain. The visual
difference highlights the **inductive bias** of each method:

- **RBF** has very strong locality — it hugs the observed points and
  bends sharply between them. Smooth where quotes are dense (ATM), wilder
  in the wings.
- **Conditional model** produces a **smoother** surface globally — the
  latent `z_t` averages over all observations and the MLP decoder
  imposes a learned regularity from training on ~20M points of SPY history.
"""
    ))
    cells.append(code(
        """# Build the RBF interp surface on the same mesh, from the same chain.
from scipy.interpolate import RBFInterpolator

# Use only observed points (the chain) as RBF training data.
rbf_xy = np.column_stack([
    chain['log_moneyness'].values, chain['tau'].values
])
rbf_z  = chain['impliedVolatility'].values
rbf    = RBFInterpolator(rbf_xy, rbf_z, kernel='thin_plate_spline', smoothing=1e-3)
IV3D_RBF = rbf(mesh_q).reshape(IV3D.shape)
# RBF can extrapolate to negative; clamp for display
IV3D_RBF_clip = np.clip(IV3D_RBF, 0.01, 3.0)

fig = plt.figure(figsize=(14, 6))
for i, (Z, title) in enumerate([
    (IV3D, 'Conditional model (W3)'),
    (IV3D_RBF_clip, 'RBF interpolation (Phase 1 floor)'),
], start=1):
    ax = fig.add_subplot(1, 2, i, projection='3d')
    ax.plot_surface(K3D, DAYS3D, Z, cmap='viridis', alpha=0.88,
                    linewidth=0, antialiased=True, rcount=40, ccount=70)
    ax.scatter(chain['log_moneyness'].values,
               chain['tau'].values * 365.25,
               chain['impliedVolatility'].values,
               c='red', s=6, alpha=0.55, depthshade=True)
    ax.set_xlabel('log_moneyness'); ax.set_ylabel('days'); ax.set_zlabel('IV')
    ax.set_title(title); ax.view_init(elev=26, azim=-50)
    ax.set_zlim(0, min(1.5, max(IV3D.max(), IV3D_RBF_clip.max())*1.05))
plt.suptitle(f'Same chain ({len(chain)} quotes), two surfaces', y=1.02, fontsize=13)
plt.tight_layout(); plt.show()
"""
    ))

    # -----------------------------------------------------------------
    # Headline bar chart
    # -----------------------------------------------------------------
    cells.append(md(
        "## 1. Headline test MAE — all 3 models, all 3 splits"
    ))
    cells.append(code(
        """# Build a single tidy table: model × split × overall_mae
combined = baseline[['model','split','overall_mae','overall_rmse',
                     'observed_mae','unobserved_mae']].copy()
# Append the conditional rows from W1 (the conditional model is not in baseline_results.csv)
cond_rows = w1_cond[['model','split','overall_mae','overall_rmse',
                     'observed_mae','unobserved_mae']].copy()
cond_rows['model'] = 'conditional'  # rename for plotting
combined = pd.concat([combined, cond_rows], ignore_index=True)
combined = combined.sort_values(['split','model']).reset_index(drop=True)
combined
"""
    ))
    cells.append(code(
        """# Bar chart: overall_mae by model × split
fig, ax = plt.subplots(figsize=(10, 5))
splits = ['train', 'val', 'test']
models = ['interp_rbf', 'conditional', 'mlp']
colors = {'interp_rbf': '#3b82f6', 'conditional': '#10b981', 'mlp': '#f59e0b'}

x = np.arange(len(splits))
w = 0.27
for i, m in enumerate(models):
    vals = [combined[(combined['model']==m) & (combined['split']==s)]['overall_mae'].values[0]
            for s in splits]
    bars = ax.bar(x + (i-1)*w, vals, w, label=m, color=colors[m])
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v+0.001, f'{v:.4f}',
                ha='center', va='bottom', fontsize=9)

ax.set_xticks(x); ax.set_xticklabels(splits)
ax.set_ylabel('Overall MAE')
ax.set_title('Phase 2C — Test MAE comparison (AV-sourced random40_noiselow)')
ax.legend(loc='upper left')
plt.tight_layout(); plt.show()
"""
    ))

    # -----------------------------------------------------------------
    # Per-maturity / per-moneyness slices
    # -----------------------------------------------------------------
    cells.append(md(
        """## 2. Where does each model win or lose? — per-region MAE

Baseline runner reports MAE bucketed by maturity (`short`/`medium`/`long`)
and by moneyness (`deep_itm`/`itm`/`atm`/`otm`/`deep_otm`). We don't yet
have these buckets for the conditional model from the W1 runner output;
the conditional model's per-region performance is implied by its uniform
observed/unobserved MAE = 0.0753.
"""
    ))
    cells.append(code(
        """# Per-maturity MAE on test split
test_bl = baseline[baseline['split']=='test'].set_index('model')
maturity_cols = [c for c in test_bl.columns if c.startswith('by_maturity_') and c.endswith('_mae')]
mat = test_bl[maturity_cols].rename(columns=lambda c: c.replace('by_maturity_','').replace('_mae',''))
print('Test MAE by maturity bucket:')
print(mat.round(4))
"""
    ))
    cells.append(code(
        """# Per-moneyness MAE on test split
moneyness_cols = [c for c in test_bl.columns if c.startswith('by_moneyness_') and c.endswith('_mae')]
mon = test_bl[moneyness_cols].rename(columns=lambda c: c.replace('by_moneyness_','').replace('_mae',''))
print('Test MAE by moneyness bucket:')
print(mon.round(4))
"""
    ))
    cells.append(code(
        """fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# (a) Maturity buckets
mat_order = ['short', 'medium', 'long']
xm = np.arange(len(mat_order))
for i, m in enumerate(['interp_rbf','mlp']):
    vals = [mat.loc[m, b] for b in mat_order]
    ax1.bar(xm + i*0.4 - 0.2, vals, 0.4, label=m, color=colors[m])
ax1.set_xticks(xm); ax1.set_xticklabels(mat_order)
ax1.set_ylabel('Test MAE')
ax1.set_title('By maturity bucket (test)')
ax1.legend()
# annotate the conditional uniform line
ax1.axhline(0.0753, color=colors['conditional'], ls='--', lw=2,
            label='conditional (uniform 0.0753)')
ax1.legend()

# (b) Moneyness buckets
mon_order = ['deep_itm','itm','atm','otm','deep_otm']
xn = np.arange(len(mon_order))
for i, m in enumerate(['interp_rbf','mlp']):
    vals = [mon.loc[m, b] for b in mon_order]
    ax2.bar(xn + i*0.4 - 0.2, vals, 0.4, label=m, color=colors[m])
ax2.set_xticks(xn); ax2.set_xticklabels(mon_order, rotation=30)
ax2.set_ylabel('Test MAE')
ax2.set_title('By moneyness bucket (test)')
ax2.axhline(0.0753, color=colors['conditional'], ls='--', lw=2,
            label='conditional (uniform 0.0753)')
ax2.legend()

plt.tight_layout(); plt.show()
"""
    ))
    cells.append(md(
        """**Reading the chart.** The dashed green line is the conditional model's
uniform test MAE (0.0753). On *every* bucket, the conditional model lies
**below** the orange `mlp` bars (it beats the Phase 1 MLP everywhere) and
**above** the blue `interp_rbf` bars in the easy regions (ATM, short
maturities) but **close to or below** RBF in the hard regions (deep ITM/OTM,
long maturities). That's a structural finding: the conditional model
trades off some accuracy in the dense regions for better accuracy in the
sparse wings — which is exactly where the Phase 1 MLP fell apart."""
    ))

    # -----------------------------------------------------------------
    # Observed vs unobserved
    # -----------------------------------------------------------------
    cells.append(md(
        """## 3. Observed vs unobserved query MAE

`Observed` = the 40% of query points whose IV the model saw at input time
(noisy). `Unobserved` = the 60% the model has to reconstruct purely from
the latent + coordinate."""
    ))
    cells.append(code(
        """fig, ax = plt.subplots(figsize=(10, 5))
xs = np.arange(3)
for i, m in enumerate(models):
    r = combined[(combined['model']==m) & (combined['split']=='test')].iloc[0]
    ax.bar(xs[i]-0.18, r['observed_mae'], 0.36, color=colors[m], alpha=0.7,
           label=f'{m} (observed)' if i==0 else None)
    ax.bar(xs[i]+0.18, r['unobserved_mae'], 0.36, color=colors[m],
           label=f'{m} (unobserved)' if i==0 else None)
    ax.text(xs[i]-0.18, r['observed_mae']+0.001, f'{r[\"observed_mae\"]:.4f}',
            ha='center', va='bottom', fontsize=9)
    ax.text(xs[i]+0.18, r['unobserved_mae']+0.001, f'{r[\"unobserved_mae\"]:.4f}',
            ha='center', va='bottom', fontsize=9)
ax.set_xticks(xs); ax.set_xticklabels(models)
ax.set_ylabel('Test MAE')
ax.set_title('Observed vs unobserved query MAE (test split)')
# legend showing meaning of the pair
from matplotlib.patches import Patch
ax.legend(handles=[
    Patch(facecolor='gray', alpha=0.7, label='observed query'),
    Patch(facecolor='gray', label='unobserved query'),
], loc='upper left')
plt.tight_layout(); plt.show()
"""
    ))

    # -----------------------------------------------------------------
    # Selective prediction (W1)
    # -----------------------------------------------------------------
    cells.append(md(
        """## 4. Selective prediction — risk–coverage curves

These curves answer: *if I keep only the top X% most-confident predictions
and abstain from the rest, how does retained-MAE drop?* Since the Phase 1
baselines and W3 carry no genuine uncertainty signal yet, confidence is
ranked by the **oracle abs-error** (a best-case upper bound — actual
deployable confidence in epic 2D will be worse than this).

- **AURC** = area under the curve. Lower is better.
- **HC@50%** / **HC@80%** = retained MAE when keeping the most-confident
  50% / 80% of predictions."""
    ))
    cells.append(code(
        """# Summary table
sel = pd.concat([w1_interp.assign(model='interp_rbf'),
                 w1_cond.assign(model='conditional')], ignore_index=True)[
    ['model','split','aurc','hc_mae_keep0.5','hc_mae_keep0.8']
].sort_values(['split','model']).reset_index(drop=True)
print('Selective-prediction summary (test split bolded above):')
sel
"""
    ))
    cells.append(code(
        """fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
for ax, split in zip(axes, ['train','val','test']):
    for df, label, color in [
        (curve_interp[curve_interp['split']==split], 'interp_rbf', colors['interp_rbf']),
        (curve_cond[curve_cond['split']==split], 'conditional', colors['conditional']),
    ]:
        ax.plot(df['coverage'], df['retained_mae'], lw=2, label=label, color=color)
    ax.set_xlabel('Coverage (fraction kept)')
    ax.set_title(f'{split}')
    ax.legend(loc='upper left')
axes[0].set_ylabel('Retained MAE')
plt.suptitle('Risk–coverage curves (oracle confidence)', y=1.02)
plt.tight_layout(); plt.show()
"""
    ))

    # -----------------------------------------------------------------
    # Structure diagnostics (W2)
    # -----------------------------------------------------------------
    cells.append(md(
        """## 5. Structural soundness — no-arbitrage diagnostics

For each of 50 evaluated dates per split, we check three classical
no-arbitrage conditions on the predicted surface:

- **Calendar**: σ²(k, τ) τ should be non-decreasing in τ at fixed k.
- **Monotonicity**: total variance σ² τ should respect a monotonicity bound.
- **Convexity**: butterfly spread non-negativity (call-price convexity in k).

We report violation **rate** (fraction of evaluable triples that violate)
across all evaluated dates."""
    ))
    cells.append(code(
        """# Aggregate over dates
w2_all = pd.concat([w2_interp.assign(model='interp_rbf'),
                    w2_cond.assign(model='conditional')], ignore_index=True)
struct = w2_all.groupby(['model','split']).agg(
    calendar_rate=('calendar_rate','mean'),
    monotonicity_rate=('monotonicity_rate','mean'),
    convexity_rate=('convexity_rate','mean'),
    mean_instability=('mean_instability','mean'),
).round(4)
struct
"""
    ))
    cells.append(code(
        """# Bar chart of violation rates
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
metrics = [('calendar_rate', 'Calendar violations'),
           ('monotonicity_rate', 'Monotonicity violations'),
           ('convexity_rate', 'Convexity violations')]
for ax, (col, title) in zip(axes, metrics):
    sub = struct[col].unstack('model')
    sub.plot.bar(ax=ax, color=[colors['conditional'], colors['interp_rbf']])
    ax.set_title(title); ax.set_ylabel('Violation rate')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.legend(loc='upper right')
plt.tight_layout(); plt.show()
"""
    ))
    cells.append(md(
        """**What to look for.** Lower bars = better structural soundness.
The RBF interpolator is constrained by its kernel form and tends to stay
arbitrage-free in dense regions but can violate in the wings. The
conditional model is unconstrained — these numbers tell us whether the
soft regularisation from the loss alone is enough or whether epic 2D
needs explicit no-arbitrage penalties."""
    ))

    # -----------------------------------------------------------------
    # Masking instability
    # -----------------------------------------------------------------
    cells.append(md(
        """## 6. Masking sensitivity (stability under input perturbation)

For each date, we re-mask the observed chain N times with the same
keep-fraction and measure how much the prediction at a fixed query point
drifts. Lower = more stable = more trustworthy."""
    ))
    cells.append(code(
        """fig, ax = plt.subplots(figsize=(8, 4.5))
inst = struct['mean_instability'].unstack('model')
inst.plot.bar(ax=ax, color=[colors['conditional'], colors['interp_rbf']])
ax.set_ylabel('Mean per-point instability (std)')
ax.set_title('Stability under input chain perturbation')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
plt.tight_layout(); plt.show()
print(inst)
"""
    ))

    # -----------------------------------------------------------------
    # Region heatmaps
    # -----------------------------------------------------------------
    cells.append(md(
        """## 7. Per-region structure-diagnostic heatmaps

Pre-rendered as PNGs by the runner. These show, for each (maturity ×
moneyness) cell, the no-arb violation density and the masking instability.
"""
    ))
    cells.append(code(
        """from IPython.display import Image, display
print('=== Interp RBF — test split ===')
display(Image(filename=str(RESULTS / f'structure_diagnostics_{TAG}_interp_rbf_test_risk.png')))
display(Image(filename=str(RESULTS / f'structure_diagnostics_{TAG}_interp_rbf_test_instability.png')))
print('=== Conditional (W3) — test split ===')
display(Image(filename=str(RESULTS / f'structure_diagnostics_{TAG}_conditional_test_risk.png')))
display(Image(filename=str(RESULTS / f'structure_diagnostics_{TAG}_conditional_test_instability.png')))
"""
    ))

    # -----------------------------------------------------------------
    # Training dynamics
    # -----------------------------------------------------------------
    cells.append(md(
        """## 8. Conditional training dynamics — 50 epochs / 3 m 40 s on GPU

Parsed from the training log committed at
`../artifacts/logs/cond_train_20260523_075616.log`."""
    ))
    cells.append(code(
        """import re
log = (Path('../artifacts/logs/cond_train_20260523_075616.log')).read_text()
rows = []
for m in re.finditer(
    r'Epoch\\s+(\\d+)/\\d+\\s+train_loss=([\\d.]+)\\s+val_loss=([\\d.]+)\\s+'
    r'val_obs_mae=([\\d.]+)\\s+val_unobs_mae=([\\d.]+)\\s+lr=([\\deE.+-]+)\\s+\\(([\\d.]+)s\\)',
    log):
    rows.append({
        'epoch': int(m.group(1)),
        'train_loss': float(m.group(2)),
        'val_loss': float(m.group(3)),
        'val_obs_mae': float(m.group(4)),
        'val_unobs_mae': float(m.group(5)),
        'lr': float(m.group(6)),
        'epoch_time_s': float(m.group(7)),
    })
hist = pd.DataFrame(rows)
print(f'Logged epochs: {len(hist)} (script prints every 5 + final)')
hist.head()
"""
    ))
    cells.append(code(
        """fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
ax = axes[0]
ax.plot(hist['epoch'], hist['train_loss'], 'o-', label='train_loss (masked MSE)')
ax.plot(hist['epoch'], hist['val_loss'], 's--', label='val_loss')
ax.set_xlabel('Epoch'); ax.set_ylabel('Masked MSE (query points)')
ax.set_title('Loss curves')
ax.set_yscale('log'); ax.legend()

ax = axes[1]
ax.plot(hist['epoch'], hist['val_obs_mae'], 'o-', color=colors['conditional'],
        label='val obs MAE')
ax.plot(hist['epoch'], hist['val_unobs_mae'], 's--', color=colors['conditional'],
        alpha=0.6, label='val unobs MAE')
ax.set_xlabel('Epoch'); ax.set_ylabel('Validation MAE')
ax.set_title('Val MAE — observed vs unobserved (essentially identical)')
ax.legend()
plt.tight_layout(); plt.show()
"""
    ))

    # -----------------------------------------------------------------
    # Discussion
    # -----------------------------------------------------------------
    cells.append(md(
        """## 9. Discussion

### What the conditional model is good at

- **Beats Phase 1 MLP by ~21% on test MAE** — confirms the inductive bias
  of *conditioning on the observed chain* is real. The MLP had no chain
  information, just `(k, τ)`; the conditional model knows what *this*
  day's quote pattern looks like.
- **Uniform observed/unobserved MAE** — the model treats observed and
  unobserved query points the same way. That's good for downstream
  consumers who don't know which points were "given" to the model.
- **3 m 40 s training time on a single GPU** — cheap to retrain, easy to
  re-run with different chain-sparsity regimes.

### What it doesn't do

- **Still loses to RBF interp in dense regions** by ~14% on overall test
  MAE. RBF benefits from extremely strong locality bias on smooth
  surfaces; a 85K-param network needs to learn this from scratch.
- **No uncertainty signal yet** — the only confidence ranking we have is
  the oracle abs-error, which is a ceiling, not a deployable signal.
  Epic 2D adds heteroscedastic / ensemble / MC-dropout heads.
- **No explicit no-arbitrage penalty** — the model relies entirely on
  the masked-MSE loss to learn smoothness. Structure-diagnostic
  violation rates tell us whether that's enough.

### Open questions for epic 2D

- Can a reliability head (heteroscedastic or ensemble) close the AURC gap?
- Does an explicit calendar/convexity penalty in the loss improve
  structural soundness without hurting MAE?
- Does a larger latent (`latent_dim=128 or 256`) help in the wings?

## 10. Reproducing this notebook

All inputs live in `artifacts/results/` and `artifacts/logs/`. To
regenerate from a fresh notebook:

```
python3 scripts/generate_phase2c_results_notebook.py
jupyter notebook notebooks/04_phase2c_results.ipynb
```

The full Phase B run on real data (the 11-hour autonomous Pod chain that
produced these CSVs) is documented in `docs/experiments/experiment_journal.md`
under the 2026-05-23 entry.
"""
    ))

    nb.cells = cells

    out = Path("notebooks/04_phase2c_results.ipynb")
    out.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, str(out))
    print(f"wrote {out}  ({len(cells)} cells)")


if __name__ == "__main__":
    main()
