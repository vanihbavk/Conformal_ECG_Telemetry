PYTHON ?= python3

.PHONY: help install test download-data verify-data benchmark unseen-noise cinc2011 cinc2017 figures reproduce clean

help:
	@echo "ecgrisk: Patient-Cluster Conformal Risk Control for ECG Quality Gating"
	@echo "Targets:"
	@echo "  make install       - Install the package in editable mode"
	@echo "  make test          - Run the unit tests and the results consistency audit (no data needed)"
	@echo "  make download-data - Download and verify all PhysioNet datasets (~370 MB) into data/"
	@echo "  make verify-data   - Verify the local datasets in data/ without downloading"
	@echo "  make benchmark     - MIT-BIH + NSTDB benchmark (47 patients, 20 runs)"
	@echo "  make unseen-noise  - Cross-regime (bw+ma -> em) shift and recalibration evaluation"
	@echo "  make cinc2011      - External validation on 998 CinC 2011 mobile-phone ECGs"
	@echo "  make cinc2017      - External validation on 8,528 CinC 2017 AliveCor ECGs"
	@echo "  make figures       - Regenerate results/figures/*.pdf from results/ and data/"
	@echo "  make reproduce     - download-data, then every evaluation above, then figures"
	@echo "  make clean         - Remove Python caches and generated LaTeX fragments"

install:
	$(PYTHON) -m pip install -e .

test:
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests -v

download-data:
	$(PYTHON) experiments/download_data.py

verify-data:
	$(PYTHON) experiments/download_data.py --verify-only

benchmark:
	PYTHONPATH=src $(PYTHON) experiments/run_benchmark.py

unseen-noise:
	PYTHONPATH=src $(PYTHON) experiments/run_unseen_noise_eval.py

cinc2011:
	PYTHONPATH=src $(PYTHON) experiments/run_cinc2011_eval.py

cinc2017:
	PYTHONPATH=src $(PYTHON) experiments/run_cinc2017_eval.py

figures:
	PYTHONPATH=src $(PYTHON) experiments/draw_all_figures.py

reproduce: download-data benchmark unseen-noise cinc2011 cinc2017 figures

clean:
	rm -rf results/latex build dist src/*.egg-info
	find . -type d -name "__pycache__" -not -path "./.git/*" -exec rm -rf {} +
	find . -type f -name "*.pyc" -not -path "./.git/*" -delete
