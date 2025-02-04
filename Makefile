.PHONY: venv install data train test run-all docker-build docker-up docker-down deploy jupyter

# Variables
VENV_NAME := venv
REQUIREMENTS := requirements.txt
SRC_DIR := src
SERVE_DIR := src/serve
ARTIFACTS_DIR := artifacts

# Environment Variables
export DATASET_VERSION := 1
export MODEL_VERSION := 1
export SAMPLING := True  # if you want to donw sample data for train and evaluation (necessary with local CPU)

# Create and activate virtual environment
venv:
	python3 -m venv $(VENV_NAME)
	@echo "Virtual environment created. To activate, run: source $(VENV_NAME)/bin/activate"

# Install requirements
install: venv
	$(VENV_NAME)/bin/pip install -r $(REQUIREMENTS)
	@echo "Requirements installed successfully"

# Run data processing step
data:
	DATASET_VERSION=$(DATASET_VERSION) \
	$(VENV_NAME)/bin/python $(SRC_DIR)/data/main.py
	@echo "Data processing completed"

# Run training step
train: data
	MODEL_VERSION=$(MODEL_VERSION) \
	SAMPLING=$(SAMPLING) \
	$(VENV_NAME)/bin/python $(SRC_DIR)/train/main.py
	@echo "Model training completed"

# Run testing step
test: train
	MODEL_VERSION=$(MODEL_VERSION) \
    SAMPLING=$(SAMPLING) \
	$(VENV_NAME)/bin/python $(SRC_DIR)/test/main.py
	@echo "Testing completed"

# Run all steps in sequence
run-all: data train test
	@echo "All steps completed successfully"

# Build Docker images
docker-build:
	cd $(SERVE_DIR) && docker-compose build
	@echo "Docker images built successfully"

# Start services
docker-up:
	cd $(SERVE_DIR) && docker-compose up -d
	@echo "Docker containers are up"

# Stop services
docker-down:
	cd $(SERVE_DIR) && docker-compose down
	@echo "Docker containers are down"

# Both build and up in one command
deploy: docker-build docker-up
	@echo "Application deployed successfully"

# Start Jupyter Notebook
jupyter:
	$(VENV_NAME)/bin/jupyter notebook --port=8888 --no-browser
	@echo "Jupyter notebook started on port 8888"