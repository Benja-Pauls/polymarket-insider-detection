# Legacy code

Market-level detection code from the initial (incorrect) framing of the
project, preserved for reference. Do not import from `pminsider` code paths.

**Why retired:** we pivoted from market-level anomaly detection to
trade-level classification with manually-curated labels from public insider-
trade callouts. The market-level approach had two fatal flaws:

1. Weak labels were derived from the same features the models saw →
   classifier AUCs were trivially ~1.0 and meaningless.
2. The "spike hit rate" analysis compared against a 50% random baseline;
   the actual baseline for "does late-stage flow agree with winner?" in
   prediction markets is far above 50% (markets converge to truth before
   resolution), so the reported signal was illusory.

The main pipeline now lives under `src/pminsider/scrape`, `src/pminsider/extract`,
and `src/pminsider/labels` — trade-level.

Data collection modules (`goldsky.py`, `polygon_rpc.py`, `collect.py`,
`scripts/enrich_catalog.py`, `scripts/refill_trades.py`) stayed in their
original locations — they operate at trade granularity and are fully reused.
