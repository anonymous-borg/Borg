format:
	uv run ruff check --fix --select=I001
	uv run ruff format
	cargo fmt
	cargo clippy
