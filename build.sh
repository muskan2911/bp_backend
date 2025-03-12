#!/usr/bin/env bash
# Exit immediately if a command exits with a non-zero status
set -o errexit  


# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # Use `venv\Scripts\activate` on Windows

# Upgrade pip to the latest version
pip install --upgrade pip  

# Install Python dependencies
pip install -r requirements.txt  

# Deactivate virtual environment (for safety)
deactivate  
