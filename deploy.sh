#!/bin/bash
echo "Starting Flask app deployment..."

# Detect OS and activate virtual environment
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

echo "Virtual environment activated."
python app.py
