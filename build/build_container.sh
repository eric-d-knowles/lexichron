#!/bin/bash
#
# Build script for ngram-kit Apptainer container
# This script uses srun to allocate a compute node to avoid OOM on login nodes.
#

set -e

DEFINITION_FILE="environment.def"
OUTPUT_IMAGE="ngram-kit.sif"

# Check if definition file exists
if [ ! -f "$DEFINITION_FILE" ]; then
    echo "Error: $DEFINITION_FILE not found"
    exit 1
fi

# Remove existing image if present
if [ -f "$OUTPUT_IMAGE" ]; then
    echo "Removing existing $OUTPUT_IMAGE..."
    rm -f "$OUTPUT_IMAGE"
fi

echo "Building Apptainer container on compute node..."
echo "This may take 15-30 minutes depending on network speed."
echo ""

# Use srun to run the build on a compute node
# Adjust --mem, --cpus-per-task, and --time as needed for your cluster
srun --mem=32G --cpus-per-task=8 --time=01:00:00 --pty \
    apptainer build --fakeroot "$OUTPUT_IMAGE" "$DEFINITION_FILE"

echo ""
echo "Build complete! Container image: $OUTPUT_IMAGE"
echo ""
echo "Test with:"
echo "  apptainer exec --nv $OUTPUT_IMAGE python -c 'import spacy; print(spacy.__version__)'"
