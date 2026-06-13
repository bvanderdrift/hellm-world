#!/bin/sh
set -e

DEST="${1:?Usage: ./setup.sh <user@host> [ssh-args...]}"
shift

ssh -t "$@" "$DEST" '
set -e

# Node.js latest LTS via NodeSource
curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
sudo apt-get install -y nodejs unzip tmux fontconfig fonts-dejavu-core libvulkan1 vulkan-tools libegl1 libopengl0 libglx0 libgles2

# pnpm
corepack enable
corepack prepare pnpm@latest --activate

# Bun
curl -fsSL https://bun.sh/install | bash
source ~/.bashrc

# Vulkan required cache folder
export XDG_RUNTIME_DIR=/run/user/$(id -u)
mkdir -p "$XDG_RUNTIME_DIR" && chmod 700 "$XDG_RUNTIME_DIR"

git clone https://github.com/bvanderdrift/hellm-world.git
cd hellm-world
pnpm i

'

ssh -t "$@" "$DEST" "cd hellm-world && exec \$SHELL -l"
