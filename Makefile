
.PHONY: setup install test format lint docker-build docker-up docker-down ingest clean

setup:
	python3 -m venv techdoc-genie-venv
	./techdoc-genie-venv/bin/activate && pip install -r requirements.txt

install:
	./techdoc-genie-venv/bin/activate && pip install -r requirements.txt

test:
	@echo "Running pytest with test environment..."
	@export PYTHONPATH=$(pwd)/src
	@export DOTENV_FILE=$(pwd)/tests/.env.test
	pytest -v tests/

format:
	black src/ tests/
	ruff check src/ tests/ --fix

lint:
	ruff check src/ tests/
	mypy src/

docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

ingest:
	python scripts/ingest_docs.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
# 	rm -rf data/processed/* data/vector_store/*

# Not sure if i want to have the last rm statementn active.  