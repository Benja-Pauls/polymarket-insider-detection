PYTHON := source ~/Documents/Personal/Stock_Portfolio/.venv/bin/activate && python
SCRIPTS := scripts

.PHONY: help data catalog trades scrape extract label match paper clean

help:
	@echo "Targets (current trade-level pipeline):"
	@echo "  catalog    Build the resolved-market catalog from Goldsky"
	@echo "  trades     Fetch per-market trade files (14d pre-resolution)"
	@echo "  scrape     Scrape Reddit/Twitter/news for insider-trade callouts"
	@echo "  extract    LLM-extract structured fields from raw callouts"
	@echo "  match      Link callouts to on-chain trade episodes"
	@echo "  label      Produce confirmed labels parquet"
	@echo "  paper      Compile paper.pdf"
	@echo "  clean      Remove generated artifacts (keeps raw cache)"

catalog:
	$(PYTHON) -m pminsider.collect --phase catalog

trades: catalog
	$(PYTHON) -m pminsider.collect --phase trades

scrape:
	$(PYTHON) $(SCRIPTS)/scrape_all.py

extract: scrape
	$(PYTHON) $(SCRIPTS)/extract_callouts.py

match: extract trades
	$(PYTHON) $(SCRIPTS)/match_callouts.py

label: match
	$(PYTHON) $(SCRIPTS)/build_trade_labels.py

paper:
	cd paper && latexmk -pdf paper.tex

clean:
	rm -rf data/processed/*.parquet data/processed/*.pkl
	rm -rf models/*.pkl models/*.joblib
	rm -rf figures/*.pdf figures/*.png
	rm -rf paper/paper.pdf paper/*.aux paper/*.log paper/*.out
