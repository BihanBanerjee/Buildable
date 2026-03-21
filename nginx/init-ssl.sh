#!/bin/bash
# ──────────────────────────────────────────────
# SSL Bootstrap Script for api.buildable.bihanbanerjee.com
# Run this ONCE on the Droplet to get the first SSL certificate
# ──────────────────────────────────────────────

DOMAIN="api.buildable.bihanbanerjee.com"
EMAIL="banerjeebihan456@gmail.com"
PROJECT_DIR="/root/Buildable"

set -e

echo "==> Step 1: Stopping any running containers..."
cd "$PROJECT_DIR"
docker compose -f docker-compose.prod.yml down 2>/dev/null || true
docker rm -f temp-nginx 2>/dev/null || true

echo "==> Step 2: Creating temporary Nginx config (HTTP only)..."

# Backup the real config
cp "$PROJECT_DIR/nginx/nginx.conf" "$PROJECT_DIR/nginx/nginx.conf.bak"

# Write a temporary HTTP-only config for Certbot challenge
cat > "$PROJECT_DIR/nginx/nginx.conf" <<'TMPCONF'
events {
    worker_connections 1024;
}
http {
    server {
        listen 80;
        server_name api.buildable.bihanbanerjee.com;

        location /.well-known/acme-challenge/ {
            root /var/www/certbot;
        }

        location / {
            return 200 'Waiting for SSL...';
            add_header Content-Type text/plain;
        }
    }
}
TMPCONF

echo "==> Step 3: Starting standalone Nginx (HTTP only)..."

# Create the shared Docker volumes if they don't exist
docker volume create buildable_certbot_webroot 2>/dev/null || true
docker volume create buildable_certbot_certs 2>/dev/null || true

# Run nginx directly — bypasses docker compose depends_on
docker run -d --name temp-nginx \
    -p 80:80 \
    -v "$PROJECT_DIR/nginx/nginx.conf:/etc/nginx/nginx.conf:ro" \
    -v buildable_certbot_webroot:/var/www/certbot \
    nginx:alpine

# Wait for nginx to be ready
sleep 3

# Verify nginx is serving on port 80
echo "==> Verifying nginx is running on port 80..."
if curl -s -o /dev/null -w "%{http_code}" http://localhost/ | grep -q "200"; then
    echo "    ✓ Nginx is serving on port 80"
else
    echo "    ✗ Nginx is NOT responding on port 80. Check 'docker logs temp-nginx'"
    exit 1
fi

echo "==> Step 4: Requesting SSL certificate from Let's Encrypt..."

docker run --rm \
    -v buildable_certbot_webroot:/var/www/certbot \
    -v buildable_certbot_certs:/etc/letsencrypt \
    certbot/certbot certonly \
    --webroot \
    --webroot-path=/var/www/certbot \
    --email "$EMAIL" \
    --agree-tos \
    --no-eff-email \
    -d "$DOMAIN"

echo "==> Step 5: Stopping temporary nginx..."
docker rm -f temp-nginx

echo "==> Step 6: Restoring full Nginx config (HTTPS)..."
cp "$PROJECT_DIR/nginx/nginx.conf.bak" "$PROJECT_DIR/nginx/nginx.conf"

echo "==> Step 7: Starting all services with docker compose..."
docker compose -f docker-compose.prod.yml up -d

echo ""
echo "==> Done! https://$DOMAIN is now live with SSL."
echo ""
echo "To auto-renew certificates, add this cron job:"
echo "0 3 * * * cd $PROJECT_DIR && docker compose -f docker-compose.prod.yml run --rm certbot renew && docker compose -f docker-compose.prod.yml exec nginx nginx -s reload"
