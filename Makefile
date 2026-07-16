VENV := .venv
BIN := $(VENV)/bin

install:
	python3 -m venv $(VENV)
	$(BIN)/pip install -U pip
	$(BIN)/pip install -r requirements-dev.txt

test:
	$(BIN)/pytest

lint:
	$(BIN)/ruff check .

coverage:
	$(BIN)/pytest --cov=. --cov-report=xml

audit:
	$(BIN)/pip-audit

build:
	docker build -t temp .

ci: lint test coverage build

.PHONY: install test lint coverage audit build ci
