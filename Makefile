.PHONY: checks, fmt, config, type, tests

# Run pre-commit checks
# ------------------------------------------------------------------------------
checks:
	uvx pre-commit run --all-files

fmt:
	uv run ruff format legow tests

lint:
	uv run ruff check --fix legow tests

type:
	uv run mypy legow --install-types --non-interactive --show-traceback

tests:
	uv run pytest --cov=legow --cov-report=term-missing tests/ -s -vv

config: ## Configure .netrc with Owkin's PyPi credentials
	$(eval PYPI_USERNAME ?= $(if $(PYPI_USERNAME),$(PYPI_USERNAME),$(shell bash -c 'read -p "PyPi Username: " username; echo $$username')))
	$(eval PYPI_PASSWORD ?= $(if $(PYPI_PASSWORD),$(PYPI_PASSWORD),$(shell bash -c 'read -s -p "PyPi Password: " pwd; echo $$pwd')))
	@if [ -z "$(PYPI_USERNAME)" ] || [ -z "$(PYPI_PASSWORD)" ]; then \
		echo "Error: PYPI_USERNAME and PYPI_PASSWORD must be set"; \
		exit 1; \
	fi
	@if [ ! -f ~/.netrc ]; then touch ~/.netrc; fi
	@chmod 600 ~/.netrc
	@if grep -q "machine pypi.owkin.com" ~/.netrc; then \
		sed -i "/machine pypi.owkin.com/,+2d" ~/.netrc; \
	fi
	@echo "machine pypi.owkin.com" >> ~/.netrc
	@echo "login $(PYPI_USERNAME)" >> ~/.netrc
	@echo "password $(PYPI_PASSWORD)" >> ~/.netrc
