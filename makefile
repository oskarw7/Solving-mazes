run: check
	@echo "Running the main script..."
	@python3 main.py

maze:
	@echo "Testing maze generation..."
	@python3 maze.py


check:
	ruff check

format:
	ruff format

clean:
	ruff clean

