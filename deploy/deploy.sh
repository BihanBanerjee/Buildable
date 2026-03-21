#!/bin/bash
# =============================================================================
# Buildable API — Deploy Script (run from local machine)
# Pulls latest code, rebuilds Docker image, restarts the container
# Usage: ./deploy/deploy.sh <droplet-ip>
# =============================================================================

set -euo pipefail

if [ -z "${1:-}" ]; then
    echo "Usage: ./deploy/deploy.sh <droplet-ip>"
    exit 1
fi

SERVER="root@$1"

echo "=== Deploying Buildable API to $SERVER ==="

ssh "$SERVER" << 'REMOTE'
set -euo pipefail
cd /opt/buildable

echo "Pulling latest code..."
git pull

echo "Rebuilding Docker image..."
docker build -t buildable-api .

echo "Restarting container..."
docker stop buildable-api 2>/dev/null || true
docker rm buildable-api 2>/dev/null || true

docker run -d \
    --name buildable-api \
    --restart unless-stopped \
    --env-file .env \
    -p 8000:8000 \
    -v /opt/buildable/projects:/app/projects \
    buildable-api

echo "Running migrations..."
docker exec buildable-api uv run alembic upgrade head

echo ""
echo "=== Deployed ==="
docker ps --filter name=buildable-api --format 'Status: {{.Status}}'
REMOTE

echo "Done! API should be live at https://$(ssh $SERVER 'cat /etc/caddy/Caddyfile | head -1 | tr -d " {"')"
