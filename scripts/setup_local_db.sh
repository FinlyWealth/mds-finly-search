#!/bin/bash

# Exit on error
set -e

# Source environment variables from .env file
if [ -f .env ]; then
    source .env
else
    echo "Error: .env file not found"
    exit 1
fi

# Check if required environment variables are set
if [ -z "$PGUSER" ] || [ -z "$PGPASSWORD" ] || [ -z "$PGHOST" ] || [ -z "$PGPORT" ] || [ -z "$PGDATABASE" ]; then
    echo "Error: Required database environment variables are not set in .env file"
    exit 1
fi

echo "Setting up database..."

# Create database user and database
psql -U postgres << EOF
CREATE USER "$PGUSER" WITH PASSWORD '$PGPASSWORD';
ALTER USER "$PGUSER" CREATEDB;
CREATE DATABASE "$PGDATABASE" OWNER "$PGUSER";
GRANT ALL PRIVILEGES ON DATABASE "$PGDATABASE" TO "$PGUSER";
EOF

# Add pgvector extension
psql -U postgres -d "$PGDATABASE" << EOF
CREATE EXTENSION IF NOT EXISTS vector;
EOF

echo "Database setup complete!"

# Run make commands
echo "Running database initialization..."
make db-load

echo "Setup complete! You can now start the application." 