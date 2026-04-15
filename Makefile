.PHONY: install install-dev test lint format clean run-local stop-local verify

install:
	pip3 install -r requirements.txt

install-dev:
	pip3 install -r requirements.txt -r requirements-dev.txt

test:
	python3 -m pytest tests/ -v --cov=src --cov-report=term-missing

lint:
	python3 -m ruff check src/ tests/
	python3 -m mypy src/ --ignore-missing-imports

format:
	python3 -m black src/ tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache mlruns/

run-local:
	docker compose up -d

stop-local:
	docker compose down

verify:
	python3 -c "from src.utils.logger import get_logger; l = get_logger('test'); l.info('Scaffold OK — Python 3.14 ready')"
