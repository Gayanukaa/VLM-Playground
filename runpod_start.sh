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
cd VLM-Playground

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
pip install -r requirements.txt

echo "✅ Environment setup complete."
