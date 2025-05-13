run: check
	@echo "Running the main script..."
	@python3 main.py

check:
	ruff check

format:
	ruff format

clean:
	ruff clean
