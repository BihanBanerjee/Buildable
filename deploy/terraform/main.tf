terraform {
  required_providers {
    digitalocean = {
      source  = "digitalocean/digitalocean"
      version = "~> 2.0"
    }
  }
}

provider "digitalocean" {
  token = var.do_token
}

# ---------------------------------------------------------------------------
# Droplet — $4/mo (512 MB RAM, 1 vCPU, 10 GB SSD)
# ---------------------------------------------------------------------------
resource "digitalocean_droplet" "buildable" {
  name     = "buildable-api"
  image    = "ubuntu-24-04-x64"
  size     = "s-1vcpu-512mb-10gb" # $4/mo
  region   = var.region
  ssh_keys = [var.ssh_key_fingerprint]

  user_data = templatefile("${path.module}/cloud-init.yaml", {
    domain = var.domain
  })

  tags = ["buildable", "production"]
}

# ---------------------------------------------------------------------------
# Firewall — only allow HTTP, HTTPS, SSH
# ---------------------------------------------------------------------------
resource "digitalocean_firewall" "buildable" {
  name        = "buildable-fw"
  droplet_ids = [digitalocean_droplet.buildable.id]

  inbound_rule {
    protocol         = "tcp"
    port_range       = "22"
    source_addresses = ["0.0.0.0/0", "::/0"]
  }

  inbound_rule {
    protocol         = "tcp"
    port_range       = "80"
    source_addresses = ["0.0.0.0/0", "::/0"]
  }

  inbound_rule {
    protocol         = "tcp"
    port_range       = "443"
    source_addresses = ["0.0.0.0/0", "::/0"]
  }

  outbound_rule {
    protocol              = "tcp"
    port_range            = "1-65535"
    destination_addresses = ["0.0.0.0/0", "::/0"]
  }

  outbound_rule {
    protocol              = "udp"
    port_range            = "1-65535"
    destination_addresses = ["0.0.0.0/0", "::/0"]
  }

  outbound_rule {
    protocol              = "icmp"
    destination_addresses = ["0.0.0.0/0", "::/0"]
  }
}

# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------
output "droplet_ip" {
  value       = digitalocean_droplet.buildable.ipv4_address
  description = "Public IP of the Buildable API server"
}

output "ssh_command" {
  value       = "ssh root@${digitalocean_droplet.buildable.ipv4_address}"
  description = "SSH into the server"
}
