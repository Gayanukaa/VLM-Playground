#!/bin/bash

# Upgrade pip
python -m pip install --upgrade pip

# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip again inside venv
python -m pip install --upgrade pip

# Clone the repo
git clone https://github.com/Gayanukaa/VLM-Playground.git

# Move into the backend directory
cd VLM-Playground/experiments

# List all branches
git branch -a

# Ask user which branch to checkout
read -p "Enter the branch you want to checkout (default 'dev'): " branch
branch=${branch:-dev}

# Checkout and update
git checkout $branch
git fetch
git pull

# Install dependencies
pip install -r requirements_with_versions.txt

echo "✅ Environment setup complete."

# Ask if user wants to start the backend
read -p "🚀 Do you want to start the backend server now? (Y/N): " run_backend

# Convert to uppercase just in case
run_backend=${run_backend^^}

if [ "$run_backend" = "Y" ]; then
    echo "🎯 Starting backend server..."
    cd jac-vision/src/backend/
    uvicorn main:app --host 0.0.0.0 --port 4000
else
    echo "❌ Backend server not started. Setup finished."
fi
