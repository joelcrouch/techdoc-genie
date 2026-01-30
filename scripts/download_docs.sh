#!/bin/bash

# Create data directories
mkdir -p data/raw/postgresql

# Download PostgreSQL 16 docs (HTML version for easier parsing)
# wget -r -l 1 -np -nd -P data/raw/postgresql \
#   https://www.postgresql.org/docs/16/

# Alternative: Download PDF version
wget -O data/raw/postgresql/postgresql-16-A4.pdf \
  https://www.postgresql.org/files/documentation/pdf/16/postgresql-16-A4.pdf

echo "Documentation downloaded to data/raw/postgresql/"