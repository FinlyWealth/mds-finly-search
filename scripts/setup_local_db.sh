#!/bin/bash

# Exit on error
set -e

# Load .env file if it exists
if [ -f .env ]; then
  echo "Loading environment variables from .env..."
  set -a
  source .env
  set +a
fi

echo "Setting up database..."

# Ensure required env vars are set
if [[ -z "$PGUSER" || -z "$PGPASSWORD" || -z "$PGDATABASE" ]]; then
  echo "ERROR: PGUSER, PGPASSWORD, or PGDATABASE is not set. Aborting."
  exit 1
fi

# Determine Postgres superuser
PG_SUPERUSER=${PG_SUPERUSER:-$(whoami)}

# Export PGPASSWORD for superuser connection (optional but safer)
export PGPASSWORD=$PGPASSWORD

# Create user if not exists
psql -U "$PG_SUPERUSER" -d postgres << EOF
DO \$\$
BEGIN
   IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = '$PGUSER') THEN
      CREATE ROLE "$PGUSER" WITH LOGIN PASSWORD '$PGPASSWORD';
      ALTER ROLE "$PGUSER" CREATEDB;
      RAISE NOTICE 'Created user $PGUSER';
   ELSE
      RAISE NOTICE 'User $PGUSER already exists, skipping';
   END IF;
END
\$\$;
EOF

# Create database if not exists
DB_EXISTS=$(psql -U "$PG_SUPERUSER" -d postgres -tAc "SELECT 1 FROM pg_database WHERE datname = '$PGDATABASE'")

if [[ -z "$DB_EXISTS" ]]; then
  echo "Creating database $PGDATABASE..."
  psql -U "$PG_SUPERUSER" -d postgres -c "CREATE DATABASE \"$PGDATABASE\" OWNER \"$PGUSER\";"
else
  echo "Database $PGDATABASE already exists, skipping creation."
fi

# Grant privileges (safe to run repeatedly)
psql -U "$PG_SUPERUSER" -d postgres -c "GRANT ALL PRIVILEGES ON DATABASE \"$PGDATABASE\" TO \"$PGUSER\";"

# Add pgvector extension (safe)
psql -U "$PG_SUPERUSER" -d "$PGDATABASE" << EOF
CREATE EXTENSION IF NOT EXISTS vector;
EOF

echo "Setup complete! You can now start the application."