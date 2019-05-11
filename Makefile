PROJECT = cv-blueprints
PYTHON_VERSION=3.7
venv = .py${PYTHON_VERSION}-${PROJECT}

# Commands that activate and run virtual environment versions.
_python = . ${venv}/bin/activate; python
_pip = . ${venv}/bin/activate; pip

default: update_venv
.PHONY: default


${venv}: requirements.txt
	python${PYTHON_VERSION} -m venv ${venv}
	${_pip} install -r requirements.txt --cache .tmp/


update_venv: requirements.txt ${venv}
	${_pip} install -r requirements.txt --cache .tmp/
	@echo Success, to activate the development environment, run:
	@echo "\tsource ${venv}/bin/activate"
.PHONY: update_venv
