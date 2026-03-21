#!/bin/bash
# =============================================================================
# Buildable API — Server Setup Script
# Run this on the DigitalOcean droplet after Terraform provisioning
# Usage: ssh root@<droplet-ip> 'bash -s' < deploy/setup.sh
# =============================================================================

set -euo pipefail

APP_DIR="/opt/buildable"
REPO_URL="https://github.com/bihanbanerjee/Buildable.git"  # Update if different

echo "=== Setting up Buildable API ==="

# Clone or pull the repo
if [ -d "$APP_DIR/.git" ]; then
    echo "Repo exists, pulling latest..."
    cd "$APP_DIR" && git pull
else
    echo "Cloning repo..."
    git clone "$REPO_URL" "$APP_DIR"
fi

cd "$APP_DIR"

# Check .env exists
if [ ! -f .env ]; then
    echo "ERROR: .env file not found!"
    echo "Copy .env to the server first:"
    echo "  scp .env.production root@<droplet-ip>:/opt/buildable/.env"
    exit 1
fi

# Build and start with Docker
echo "Building Docker image..."
docker build -t buildable-api .

# Stop existing container if running
docker stop buildable-api 2>/dev/null || true
docker rm buildable-api 2>/dev/null || true

# Run the container
echo "Starting Buildable API..."
docker run -d \
    --name buildable-api \
    --restart unless-stopped \
    --env-file .env \
    -p 8000:8000 \
    -v /opt/buildable/projects:/app/projects \
    buildable-api

# Run database migrations
echo "Running database migrations..."
docker exec buildable-api uv run alembic upgrade head

echo ""
echo "=== Buildable API is running ==="
echo "Container: $(docker ps --filter name=buildable-api --format '{{.Status}}')"
echo "Logs:      docker logs -f buildable-api"
