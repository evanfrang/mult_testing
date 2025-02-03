#!/bin/bash

# Define the Quarto input file and Python script
QUARTO_FILE="README.qmd"  # Replace with your Quarto file path
PYTHON_SCRIPT="folder.py"  # Replace with your Python script path

# Check if Quarto file exists
if [ ! -f "$QUARTO_FILE" ]; then
    echo "Error: Quarto file '$QUARTO_FILE' not found!"
    exit 1
fi

# Check if Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Python script '$PYTHON_SCRIPT' not found!"
    exit 1
fi

# Step 1: Render the Quarto file
echo "Rendering Quarto file: $QUARTO_FILE..."
quarto render "$QUARTO_FILE"

# Check if Quarto rendering was successful
if [ $? -ne 0 ]; then
    echo "Error: Quarto rendering failed!"
    exit 1
fi
echo "Quarto rendering completed successfully."

# Step 2: Run the Python script
echo "Running Python script: $PYTHON_SCRIPT..."
python "$PYTHON_SCRIPT"  # Use `python` if Python 3 is your default

# Check if Python script execution was successful
if [ $? -ne 0 ]; then
    echo "Error: Python script execution failed!"
    exit 1
fi
echo "Python script executed successfully."