#!/usr/bin/env bash
# Build script for Render

set -o errexit

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

echo "Build completed successfully!"
