# Shortcuts. On Windows without `make`, run the command on the right-hand side directly
# (see README "Windows" section).

PY ?= python

.PHONY: install eda train test lint serve docker-build docker-run all clean

install:          ## install the package + API + dev tools
	$(PY) -m pip install -e ".[api,dev]"

eda:              ## exploratory analysis -> reports/eda_summary.md + figures
	$(PY) -m wdbc.eda

train:            ## cross-validate, pick model + threshold, test once, save model
	$(PY) -m wdbc.train

test:             ## unit + end-to-end tests
	$(PY) -m pytest -q

lint:             ## style and bug checks
	ruff check src tests && ruff format --check src tests

serve:            ## run the API at http://127.0.0.1:8000/docs
	uvicorn wdbc.api:app --reload

docker-build:
	docker build -t wdbc-api .

docker-run:
	docker run --rm -p 8000:8000 wdbc-api

all: lint test eda train

clean:
	rm -rf models/*.joblib models/*.json reports/runs .pytest_cache mlruns
