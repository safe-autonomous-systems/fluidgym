SRC_DIR=src/fluidgym
PICT_DIR=src/fluidgym/simulation/pict
EXAMPLES_DIR=examples
TEST_DIR=tests
COVERAGE_REPORT=term-missing

PYTHON ?= python
PYTEST ?= python -m pytest
PIP ?= python -m pip
MAKE ?= make
RUFF ?= ruff
MYPY ?= mypy
PRECOMMIT ?= pre-commit

.PHONY: check-ruff
check-ruff:
	$(RUFF) check ${SRC_DIR} --fix || :
	$(RUFF) check ${EXAMPLES_DIR} --fix || :
	$(RUFF) check ${TESTS_DIR} --fix || :

.PHONY: check-mypy
check-mypy:
	$(MYPY) ${SRC_DIR} --exclude "${PICT_DIR}/*" || :

check: check-ruff check-mypy 

.PHONY: docs
docs:
	cd docs && $(MAKE) html && cd ..

.PHONY: upload-docs
upload-docs: docs
	ghp-import -n -p -f docs/build/html

.PHONY: pre-commit
pre-commit:
	$(PRECOMMIT) run --all-files || :

.PHONY: format
format:
	isort ${SRC_DIR}
	ruff format

.PHONY: test
test:
	pytest --cov=$(SRC_DIR) --cov-report=$(COVERAGE_REPORT) $(TEST_DIR)

# HTML coverage report
.PHONY: coverage-html
coverage-html:
	pytest --cov=$(SRC_DIR) --cov-report=html $(TEST_DIR)

.PHONY: install
install: clean build
	pip install dist/fluidgym-*.whl

.PHONY: clean
clean:
	rm -rf build dist __pycache__ src/*.egg-info
	rm -rf $(SRC_DIR)/**/*.pyc $(SRC_DIR)/**/*.pyo
	rm -rf $(TEST_DIR)/**/*.pyc $(TEST_DIR)/**/*.pyo
	rm -rf docs/_build


.PHONY: build
build: clean
	MAX_JOBS=1 \
	TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0+PTX" \
	$(PYTHON) -m pip wheel . -w dist --no-build-isolation --no-deps

.PHONY: build-manylinux
build-manylinux: clean
	@CIBW_MANYLINUX_X86_64=manylinux2014 \
	CIBW_ENVIRONMENT='PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu128 TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0+PTX"' \
	CIBW_REPAIR_WHEEL_COMMAND='auditwheel repair -w {dest_dir} {wheel} --exclude "libtorch*" --exclude "libc10*"' \
	cibuildwheel --output-dir dist .