#!/bin/bash

# Exit on error
set -e

# Source environment variables from .env file if it exists
if [ -f .env ]; then
    source .env
fi

# Configuration
PROXY_DIR=".cloud_sql_proxy"
INSTANCE=${CLOUD_SQL_INSTANCE:-"pristine-flames-460002-h2:us-west1:postgres"}
PORT="5433"

# Create proxy directory if it doesn't exist
mkdir -p "$PROXY_DIR"

# Determine OS and architecture
OS=$(uname -s)
ARCH=$(uname -m)

# Set the appropriate binary URL
if [ "$OS" = "Darwin" ]; then
    if [ "$ARCH" = "arm64" ]; then
        BINARY_URL="https://dl.google.com/cloudsql/cloud_sql_proxy.darwin.arm64"
    else
        BINARY_URL="https://dl.google.com/cloudsql/cloud_sql_proxy.darwin.amd64"
    fi
elif [ "$OS" = "Linux" ]; then
    BINARY_URL="https://dl.google.com/cloudsql/cloud_sql_proxy.linux.amd64"
else
    echo "Unsupported OS: $OS"
    exit 1
fi

# Download the proxy if it doesn't exist
if [ ! -f "$PROXY_DIR/cloud_sql_proxy" ]; then
    echo "Downloading Cloud SQL proxy..."
    curl -o "$PROXY_DIR/cloud_sql_proxy" "$BINARY_URL"
    chmod +x "$PROXY_DIR/cloud_sql_proxy"
fi

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "Error: gcloud is not installed. Please install the Google Cloud SDK."
    exit 1
fi

# Check if user is authenticated
if ! gcloud auth application-default print-access-token &> /dev/null; then
    echo "Please authenticate with Google Cloud:"
    gcloud auth application-default login
fi

# Start the proxy
echo "Starting Cloud SQL proxy..."
"$PROXY_DIR/cloud_sql_proxy" -instances="$INSTANCE"=tcp:"$PORT" 