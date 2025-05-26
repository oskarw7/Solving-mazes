
run: check
	@echo "Running the main script..."
	@python3 main.py

train:
	@echo "Training the model..."
	@python3 hh_learn_test.py

maze:
	@echo "Testing maze generation..."
	@python3 maze.py


check:
	@ruff check

format:
	ruff format

clean:
	ruff clean

