# For source checking and testing

all: rufflint mypy test

rufflint:
	ruff check

rufffmt:
	-ruff check --select I --fix .
	-ruff format .

mypy:
	mypy

# Change to --cov-report html to generate html coverage reports
test:
	pytest --cov --cov-report term-missing tests/

