.PHONY: test clean structure

# Run tests
test:
	python -m pytest tests/ -vvv

# Clean build artifacts (optional but useful)
clean:
	@echo "Cleaning artifacts..."
	rm -rf build/ dist/ *.egg-info/ logs/
	find . -name "__pycache__" -type d -delete

structure:
	tree -L 4 --noreport src/dl_techniques/ | sed 's/^/ /'
