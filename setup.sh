#!/bin/sh
set -e

HOST="${1:?Usage: ./setup.sh <ip-or-host>}"

ssh -t "root@$HOST" '
set -e

# Node.js latest LTS via NodeSource
curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
sudo apt-get install -y nodejs unzip tmux

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

ssh -t "root@$HOST" "cd hellm-world && exec \$SHELL -l"
