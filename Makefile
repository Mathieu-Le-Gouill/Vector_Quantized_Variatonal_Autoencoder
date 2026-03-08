# Variables
PYTHON=python3
VENV=venv
PIP=$(VENV)/bin/pip
PY=$(VENV)/bin/python

# Default target
all: help

# --- Create virtual environment and install dependencies ---
$(VENV):
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

# --- Run training (single run) ---
train: $(VENV)
	$(PY) train.py

# --- Run training with Hydra multirun (sweeps) ---
# Usage example: make sweep PARAMS="training.epochs=1,3,5 optimizer.lr=1e-3,1e-4"
sweep: $(VENV)
	$(PY) train.py --multirun $(PARAMS)

# --- Run inference / translation ---
translate: $(VENV)
	$(PY) translate.py

# --- Clean temporary files and outputs ---
clean:
	rm -rf __pycache__ $(VENV)
	find . -name "*.pyc" -delete
	find . -name "*.pt" -delete
	rm -rf outputs/*

# --- Show help ---
help:
	@echo "Makefile commands:"
	@echo "  make train                   # run single training"
	@echo "  make sweep PARAMS=\"...\"    # run Hydra multirun/sweep, e.g.:"
	@echo "                               # make sweep PARAMS=\"training.epochs=1,3,5 optimizer.lr=1e-3,1e-4\""
	@echo "  make clean                   # remove temp files, virtualenv, checkpoints, Hydra outputs"
	@echo "  make all                     # show this help"