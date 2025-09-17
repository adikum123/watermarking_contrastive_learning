#!/bin/bash

# Define the target directory
TARGET_DIR="LJSpeech"

# Create the directory if it doesn't exist
mkdir -p "$TARGET_DIR"

# Navigate into the directory
cd "$TARGET_DIR" || exit

# Download the LJSpeech dataset
wget -c https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2

# Extract the dataset
tar -xvjf LJSpeech-1.1.tar.bz2

# Clean up the tarball
rm LJSpeech-1.1.tar.bz2

echo "LJSpeech dataset has been downloaded and extracted to $TARGET_DIR"