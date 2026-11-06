#!/bin/bash
set -euxo pipefail

export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y ca-certificates curl gnupg rsync

# Swap so t3.micro (1GB RAM) can limp through Docker Compose demos
SWAP_GB="${enable_swap_gb}"
if [[ "$${SWAP_GB}" != "0" && ! -f /swapfile ]]; then
  fallocate -l "$${SWAP_GB}G" /swapfile || dd if=/dev/zero of=/swapfile bs=1M count=$((SWAP_GB * 1024))
  chmod 600 /swapfile
  mkswap /swapfile
  swapon /swapfile
  echo '/swapfile none swap sw 0 0' >> /etc/fstab
  sysctl vm.swappiness=10
fi

install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
chmod a+r /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo \"$VERSION_CODENAME\") stable" \
  > /etc/apt/sources.list.d/docker.list

apt-get update -y
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

usermod -aG docker ubuntu

mkdir -p /opt/${project_name}
chown ubuntu:ubuntu /opt/${project_name}

cat >/etc/profile.d/${project_name}.sh <<EOF
export AI_RESEARCH_APP_DIR=/opt/${project_name}
export AI_RESEARCH_UPLOADS_BUCKET=${uploads_bucket}
export AI_RESEARCH_DOMAIN=${domain_name}
EOF

# Marker so deploy scripts know bootstrap finished
touch /opt/${project_name}/.bootstrap-ready
chown ubuntu:ubuntu /opt/${project_name}/.bootstrap-ready
