#!/usr/bin/env bash
# ============================================================
# Environment Setup Script for All Devoirs (0-3)
# ============================================================
# This script sets up a unified Python environment compatible
# with all 4 devoirs.

set -e  # Exit on error

echo "🚀 Setting up NLP Devoirs Environment..."
echo "========================================"

# Check if Python 3.10+ is available
PYTHON_CMD=$(command -v python3 || command -v python)
PYTHON_VERSION=$($PYTHON_CMD --version | awk '{print $2}')
echo "✓ Python version: $PYTHON_VERSION"

# Create or use existing virtual environment
VENV_PATH=".venv"
if [ ! -d "$VENV_PATH" ]; then
    echo "📦 Creating virtual environment at $VENV_PATH..."
    $PYTHON_CMD -m venv "$VENV_PATH"
else
    echo "✓ Virtual environment already exists at $VENV_PATH"
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source "$VENV_PATH/bin/activate"

# Upgrade pip, setuptools, wheel
echo "🔨 Upgrading pip, setuptools, wheel..."
pip install --upgrade pip setuptools wheel

# Install unified requirements
echo "📥 Installing unified requirements (all devoirs)..."
pip install -r requirements.txt

# Summary
echo ""
echo "✅ Environment Setup Complete!"
echo "========================================"
echo ""
echo "📋 Summary:"
echo "  - Virtual environment: $VENV_PATH"
echo "  - Main requirements: requirements.txt"
echo "  - Per-devoir requirements:"
echo "    • devoir0/requirements.txt (Basic NLP)"
echo "    • devoir1/requirements.txt (Legal Document Pipeline)"
echo "    • devoir2/requirements.txt (RAG System)"
echo "    • devoir3/requirements.txt (Advanced RAG + Graphs)"
echo ""
echo "🎯 Next steps:"
echo "  1. Activate environment: source $VENV_PATH/bin/activate"
echo "  2. Run individual devoirs (all packages already installed)"
echo "  3. For specific devoir testing: pip install -r devoir<N>/requirements.txt"
echo ""
