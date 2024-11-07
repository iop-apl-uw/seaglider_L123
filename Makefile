# For source checking and testing

all: rufflint mypy test

rufflint:
	ruff check .

rufffmt:
	-ruff check --select I --fix *py tests/*py
	-ruff format *py tests/*py

mypy:
	mypy

# Change to --cov-report html to generate html coverage reports
test:
	pytest --cov --cov-report term-missing tests/

testhtml:
	pytest --cov --cov-report html tests/

act:
	act -j check --container-daemon-socket -  --container-architecture linux/aarch64 push
