PYTHON := source ~/Documents/Personal/Stock_Portfolio/.venv/bin/activate && python
SCRIPTS := scripts

.PHONY: help data features labels models figures paper clean check

help:
	@echo "Targets:"
	@echo "  data       Collect raw subgraph + RPC data into data/raw/"
	@echo "  features   Build feature matrix → data/processed/features.parquet"
	@echo "  labels     Assemble labeled dataset → data/processed/labels.parquet"
	@echo "  models     Train + evaluate all models → models/, results/"
	@echo "  figures    Regenerate paper figures → figures/"
	@echo "  paper      Compile paper.pdf"
	@echo "  check      Verify data pipeline integrity"
	@echo "  clean      Remove generated artifacts (keeps raw cache)"

data:
	$(PYTHON) -m pminsider.collect --all

features: data
	$(PYTHON) $(SCRIPTS)/build_features.py

labels: data
	$(PYTHON) $(SCRIPTS)/build_labels.py

models: features labels
	$(PYTHON) $(SCRIPTS)/train_models.py

figures: models
	$(PYTHON) $(SCRIPTS)/make_figures.py

paper: figures
	cd paper && latexmk -pdf paper.tex

check:
	$(PYTHON) $(SCRIPTS)/check_pipeline.py

clean:
	rm -rf data/processed/*.parquet data/processed/*.pkl
	rm -rf models/*.pkl models/*.joblib
	rm -rf figures/*.pdf figures/*.png
	rm -rf paper/paper.pdf paper/*.aux paper/*.log paper/*.out
