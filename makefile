test:
	@echo "Testing model..."
	@python3 models/test_learned.py

run: check
	@echo "Running the main script..."
	@python3 main.py

train:
	@echo "Training the model..."
	@python3 models/heuristic_model.py

maze:
	@echo "Testing maze generation..."
	@python3 utils/maze.py


check:
	@ruff check

format:
	ruff format

clean:
	ruff clean

