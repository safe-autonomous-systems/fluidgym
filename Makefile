SRC_DIR=src/fluidgym
EXAMPLES_DIR=examples
TEST_DIR=tests

PYTHON ?= python
PYTEST ?= python -m pytest
PIP ?= python -m pip
MAKE ?= make
RUFF ?= ruff
MYPY ?= mypy
PRECOMMIT ?= pre-commit
UV ?= uv

.PHONY: check-ruff
check-ruff:
	$(RUFF) check ${SRC_DIR} --fix || :
	$(RUFF) check ${EXAMPLES_DIR} --fix || :
	$(RUFF) check ${TEST_DIR} --fix || :

.PHONY: check-mypy
check-mypy:
	$(MYPY) ${SRC_DIR} || :

check: check-ruff check-mypy

# Deployed to GitHub Pages by the Docs workflow (Actions -> Docs)
.PHONY: docs
docs:
	cd docs && $(MAKE) html && cd ..

.PHONY: pre-commit
pre-commit:
	$(PRECOMMIT) run --all-files || :

.PHONY: format
format:
	$(RUFF) format

.PHONY: test
test:
	$(PYTEST) $(TEST_DIR)

.PHONY: install
install:
	$(PIP) install .

.PHONY: install-dev
# phipict (the CUDA solver) must be installed first, see
# https://github.com/safe-autonomous-systems/phiPICT
install-dev:
	$(PIP) install --group dev -e .

.PHONY: clean
clean:
	rm -rf build dist .pytest_cache
	find $(SRC_DIR) $(TEST_DIR) -name __pycache__ -prune -exec rm -rf {} +
	rm -rf docs/build

.PHONY: build
build: clean
	$(UV) build
