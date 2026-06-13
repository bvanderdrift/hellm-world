#!/bin/sh
set -e

DEST="${1:?Usage: ./setup.sh <user@host> [ssh-args...]}"
shift

ssh -t "$@" "$DEST" '
set -e

# Node.js latest LTS via NodeSource
curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
sudo apt-get install -y nodejs unzip tmux fontconfig fonts-dejavu-core

# pnpm
corepack enable
corepack prepare pnpm@latest --activate

# Bun
curl -fsSL https://bun.sh/install | bash
source ~/.bashrc

git clone https://github.com/bvanderdrift/hellm-world.git
cd hellm-world
pnpm i

'

ssh -t "$@" "$DEST" "cd hellm-world && exec \$SHELL -l"
