install:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

lint-fix:
	./tools/format.sh

install-hooks:
	pre-commit install
