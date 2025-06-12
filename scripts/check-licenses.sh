#!/bin/bash
# License compatibility check script for RSVM project
# Ensures no GPL/LGPL-only dependencies that would conflict with MIT/BSD-3-Clause licensing

set -e

echo "=== License Compatibility Check ==="
echo "Project target licenses: MIT OR BSD-3-Clause"
echo ""

# Check if cargo-license is installed
if ! command -v cargo-license &> /dev/null; then
    echo "Installing cargo-license..."
    cargo install cargo-license
fi

# Get all licenses
echo "Analyzing dependency licenses..."
LICENSES=$(cargo license)

# Display summary
echo ""
echo "License summary:"
echo "$LICENSES" | grep -E "^(MIT|Apache-2.0|BSD)" | wc -l | xargs echo "  Compatible licenses (MIT/Apache/BSD):"
echo "$LICENSES" | grep -E " OR " | wc -l | xargs echo "  Multi-license (with options):"

# Check for problematic licenses
echo ""
echo "Checking for GPL/LGPL-only dependencies..."

# Find GPL/LGPL licenses that are NOT part of multi-license (OR) packages
PROBLEMATIC=$(echo "$LICENSES" | grep -E "^(GPL|LGPL)" | grep -v " OR " || true)

if [ ! -z "$PROBLEMATIC" ]; then
    echo "❌ ERROR: Found GPL/LGPL-only dependencies:"
    echo "$PROBLEMATIC"
    echo ""
    echo "These dependencies are incompatible with MIT/BSD-3-Clause licensing."
    exit 1
fi

# Check multi-license packages with GPL/LGPL options
MULTI_LICENSE_GPL=$(echo "$LICENSES" | grep -E "(GPL|LGPL)" | grep " OR " || true)

if [ ! -z "$MULTI_LICENSE_GPL" ]; then
    echo "ℹ️  Found multi-license dependencies with GPL/LGPL options (these are OK):"
    echo "$MULTI_LICENSE_GPL"
    echo ""
    echo "These packages offer non-GPL alternatives and are acceptable."
fi

echo ""
echo "✅ License check passed! No GPL/LGPL-only dependencies found."
echo "The project maintains compatibility with MIT/BSD-3-Clause licensing."