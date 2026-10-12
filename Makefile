.PHONY: install test test-realdata lint demo benchmark data clean

install:
	pip install -e ".[dev]"

test:
	pytest

data:
	gammaguard fetch-data all

test-realdata: data
	pytest -m realdata

lint:
	ruff check src tests examples validation

demo:
	gammaguard demo && gammaguard demo --detector nai

benchmark:
	gammaguard benchmark --detector hpge --trials 100
	gammaguard benchmark --detector nai --trials 100

clean:
	rm -rf build dist .pytest_cache .ruff_cache src/*.egg-info
