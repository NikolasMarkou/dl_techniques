.PHONY: test clean structure docs

# Run tests
test:
	python -m pytest tests/ -vvv

# Clean build artifacts (optional but useful)
clean:
	@echo "Cleaning artifacts..."
	rm -rf build/ dist/ *.egg-info/ logs/
	find ./src/ -name "__pycache__" -type d -type d -exec rm -rf {} +
	find ./tests/ -name "__pycache__" -type d -type d -exec rm -rf {} +

structure:
	tree -L 4 --noreport src/dl_techniques/ | sed 's/^/ /'

docs:
	python generate_docs.py