PYTHON := python3
VENV_DIR := ".venv"
VENV_PYTHON := $(VENV_DIR)/bin/python
VENV_ACTIVATE := . $(VENV_DIR)/bin/activate


$(VENV_DIR)/bin/activate: requirements.txt
	$(PYTHON) -m venv $(VENV_DIR)
	$(VENV_PYTHON) -m pip install --upgrade pip
	$(VENV_PYTHON) -m pip install -r requirements.txt

# Define the target to install dependencies
install: $(VENV_DIR)/bin/activate

# Define the target to clean the virtual environment
clean:
	rm -rf $(VENV_DIR)

run: install
	$(VENV_ACTIVATE)