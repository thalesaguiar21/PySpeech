test:
	python -m unittest -v $(module)

# Delete every __pycache__ directory in this project
# {} + appends every directory found by find command to rm command
# -type d restricts find command to directories
clear:
	find . -type d -name '__pycache__' -exec rm -r {} +
