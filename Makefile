PYTHON := python3
SHELL = /bin/bash -o pipefail
VENV_DIR := ".venv"
VENV_PYTHON := $(VENV_DIR)/bin/python
VENV_ACTIVATE := . $(VENV_DIR)/bin/activate

venv ?= .venv
pip := $(venv)/bin/pip

$(VENV_DIR)/bin/activate: requirements.txt
	$(PYTHON) -m venv $(VENV_DIR)
	$(VENV_PYTHON) -m pip install --upgrade pip
	$(VENV_PYTHON) -m pip install -r requirements.txt

.git/hooks/pre-push: $(venv)
	$(venv)/bin/pre-commit install --install-hooks -t pre-push

# Define the target to install dependencies
install: $(venv) node_modules $(if $(value CI),,install-hooks)

node_modules: package.json
	npm install
	touch node_modules

install-hooks: .git/hooks/pre-push

# Define the target to clean the virtual environment
clean:
	rm -rf $(VENV_DIR)

run: install
	$(VENV_ACTIVATE)

check: hooks

## run pre-commit hooks on all files
hooks: node_modules $(venv)
	$(venv)/bin/pre-commit run --color=always --all-files --hook-stage push