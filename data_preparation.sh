#!/bin/bash

if command -v python3.9 &>/dev/null; then
  echo "Python 3.9 is already installed."
else
  # Add the repository for Python 3.9
  sudo add-apt-repository -y ppa:deadsnakes/ppa
  # Update package lists
  sudo apt-get update
  # Install Python 3.9
  sudo apt-get install python3.9 -y
  # Check if Python 3.9 is installed correctly
fi

if [[ -d "NLSN" ]]; then
  echo "Virtual environment 'NLSN' exists."
  source NLSN/bin/activate
else
  echo "Virtual environment 'NLSN' does not exist."
  python3.9 -m venv NLSN
  source NLSN/bin/activate
  pip install super-image
  pip install datasets
fi

python3.9 test.py