variable "do_token" {
  description = "DigitalOcean API token"
  type        = string
  sensitive   = true
}

variable "domain" {
  description = "Domain name for the API (e.g. api.buildable.dev)"
  type        = string
}

variable "region" {
  description = "DigitalOcean region"
  type        = string
  default     = "blr1" # Bangalore — closest to India
}

variable "ssh_key_fingerprint" {
  description = "SSH key fingerprint already added to DigitalOcean"
  type        = string
}
