.PHONY: clean clean-build clean-pyc clean-test coverage dist docs help install lint lint/flake8 lint/black
.DEFAULT_GOAL := help

define BROWSER_PYSCRIPT
import os, webbrowser, sys

from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

BROWSER := python -c "$$BROWSER_PYSCRIPT"

help: ## Show this help message
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

clean: clean-build clean-pyc clean-test ## Remove all build, test, coverage and Python artifacts

clean-build: ## Remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc: ## Remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## Remove test and coverage artifacts
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache

lint: lint/flake8 lint/black ## Check style with flake8 and black

lint/flake8: ## Check style with flake8
	flake8 src tests
lint/black: ## Check style with black
	black --check src tests

test: ## Run tests quickly with the default Python
	pytest

test-all: ## Run tests on every Python version with tox
	tox

coverage: ## Check code coverage quickly with the default Python
	pytest --cov=churnsense tests/
	coverage report -m
	coverage html
	$(BROWSER) htmlcov/index.html

docs: ## Generate Sphinx HTML documentation, including API docs
	rm -f docs/api/churnsense*.rst
	rm -f docs/api/modules.rst
	sphinx-apidoc -o docs/api src/churnsense
	$(MAKE) -C docs clean
	$(MAKE) -C docs html
	$(BROWSER) docs/_build/html/index.html

servedocs: docs ## Compile the docs watching for changes
	watchmedo shell-command -p '*.rst' -c '$(MAKE) -C docs html' -R -D .

release: dist ## Package and upload a release
	twine upload dist/*

dist: clean ## Builds source and wheel package
	python -m build
	ls -l dist

install: clean ## Install the package to the active Python's site-packages
	pip install -e .

develop: clean ## Install development dependencies
	pip install -e ".[dev]"

docker-build: ## Build the Docker image
	docker build -t churnsense .

docker-run: ## Run the Docker container
	docker run -p 8050:8050 churnsense

dashboard: ## Run the dashboard locally
	python dashboard.py