#!/bin/bash
# =============================================================================
# SSL Setup — Run AFTER DNS A record points to the droplet IP
# Usage: ssh root@<droplet-ip> 'bash -s' < deploy/ssl-setup.sh
# =============================================================================

set -euo pipefail

if [ -z "${1:-}" ]; then
    read -p "Enter your domain (e.g. api.buildable.dev): " DOMAIN
    read -p "Enter your email (for Let's Encrypt notices): " EMAIL
else
    DOMAIN="$1"
    EMAIL="${2:-admin@$DOMAIN}"
fi

echo "=== Setting up SSL for $DOMAIN ==="

# Get SSL certificate via Certbot
certbot --nginx \
    -d "$DOMAIN" \
    --non-interactive \
    --agree-tos \
    -m "$EMAIL" \
    --redirect  # Auto-redirect HTTP → HTTPS

# Certbot auto-renewal is installed by default via systemd timer
echo ""
echo "=== SSL configured ==="
echo "Certificate: /etc/letsencrypt/live/$DOMAIN/"
echo "Auto-renewal: systemctl status certbot.timer"
echo "Test renewal: certbot renew --dry-run"
