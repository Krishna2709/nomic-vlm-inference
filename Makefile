.PHONY: help install test lint format clean build run docker-build docker-run

# Default target
help:
	@echo "Available commands:"
	@echo "  install     - Install development dependencies"
	@echo "  test        - Run tests"
	@echo "  lint        - Run linting checks"
	@echo "  format      - Format code"
	@echo "  clean       - Clean up temporary files"
	@echo "  build       - Build Docker image"
	@echo "  run         - Run the API locally"
	@echo "  docker-run  - Run Docker container"

# Install dependencies
install:
	pip install -r requirements-dev.txt
	pre-commit install

# Run tests
test:
	pytest tests/ -v --cov=api --cov-report=html --cov-report=term

# Run linting
lint:
	flake8 api/ tests/
	mypy api/
	bandit -r api/

# Format code
format:
	black api/ tests/
	isort api/ tests/

# Clean up
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .coverage htmlcov/ .pytest_cache/ .mypy_cache/

# Build Docker image
build:
	docker build -t nomic-vlm-inference ./api/

# Run API locally
run:
	cd api && uvicorn app:app --host 0.0.0.0 --port 8000 --reload

# Run Docker container
docker-run:
	docker run --rm -p 8000:8000 nomic-vlm-inference

# Run all quality checks
check: lint test

# Pre-commit checks
pre-commit:
	pre-commit run --all-files

# Development setup
dev-setup: install
	@echo "Development environment setup complete!"
	@echo "Run 'make run' to start the API server"