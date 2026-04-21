# Polymarket Insider Detection

Research project: detecting informed trading in Polymarket prediction markets via on-chain volume and flow anomalies.

## Thesis

Large directional trades placed into previously-quiet prediction markets hours before the underlying event publicly resolves are detectable as statistical anomalies. If those flagged trades resolve in the direction of the anomaly at a rate meaningfully above the market's base rate, that constitutes evidence of informed (possibly insider) trading.

Canonical motivating case: a bettor placed a ~$400K position on Polymarket hours before the United States publicly announced the capture of Nicolás Maduro, netting hundreds of thousands in profit. The volume and flow pattern should have been detectable before the resolution.

## Approach

| Phase | Deliverable |
|-------|-------------|
| Data collection | Goldsky subgraph snapshots of all resolved Polymarket markets with per-trade granularity (wallet, side, size, price, timestamp) |
| Labeling | Strong positives from news/Twitter/Reddit callouts; weak positives from anomaly heuristics; excluded scheduled-information events |
| Feature engineering | Volume-spike ratios at multiple horizons, directional imbalance, price acceleration, wallet concentration (Herfindahl), cross-market wallet correlation |
| Modeling | Logistic regression baseline, XGBoost, LightGBM, Isolation Forest; train/val/test split *by market* to avoid leakage |
| Evaluation | ROC-AUC, PR-AUC, precision@k; compared to baseline "always guess majority" and "random" |
| Paper | LaTeX writeup in `paper/` with figures in `figures/` |

## Data sources (on-chain only — Polymarket REST APIs are geo-blocked for US IPs)

- **Goldsky hosted subgraphs** — primary source of trade-level data
  - `orderbook-subgraph/0.0.1` (live) — `orderFilledEvent` with maker/taker/sizes/prices
  - `polymarket-orderbook-resync/prod` — enriched historical `enrichedOrderFilled` with Account links
  - `activity-subgraph/0.0.4` — `Condition` entities (resolution timestamps, payouts)
  - `pnl-subgraph/0.0.14` — per-wallet aggregate statistics
  - `oi-subgraph/0.0.6` — market-level open interest
- **Polygon RPC** (publicnode, drpc, onfinality, 1rpc) — block-timestamp resolution, cross-validation

## Layout

```
src/pminsider/        # library code
  goldsky.py          # GraphQL subgraph client
  polygon_rpc.py      # JSON-RPC client with fallback pool
  collect.py          # high-level data-collection workflows
  features.py         # feature engineering
  labels.py           # labeling pipeline (news callouts + heuristics)
  models.py           # model training + evaluation harness
  viz.py              # figure generation
scripts/              # entry-point scripts (each maps to a Makefile target)
data/
  raw/goldsky/        # cached GraphQL responses (parquet)
  raw/polygon/        # cached RPC responses
  processed/          # feature matrices, labels
notebooks/            # exploratory notebooks
paper/                # LaTeX source + references.bib
figures/              # generated figures used in the paper
```

## Setup

The project reuses the Python venv at `~/Documents/Personal/Stock_Portfolio/.venv`. Activate with:

```bash
source ~/Documents/Personal/Stock_Portfolio/.venv/bin/activate
```

Dependencies installed: `pandas`, `numpy`, `scikit-learn`, `statsmodels`, `matplotlib`, `seaborn`, `xgboost`, `lightgbm`, `tqdm`, `pyarrow`, `web3`, `networkx`, `curl_cffi`, `requests`.

## Reproducibility

```bash
make data       # collect raw data from Goldsky
make features   # build feature matrix
make labels     # assemble labeled dataset
make models     # train and evaluate all models
make figures    # regenerate every figure in the paper
make paper      # compile paper.pdf
```

All data artifacts are content-hashed and versioned under `data/processed/`. The paper figures are regenerated from pinned artifact hashes so the build is reproducible.
