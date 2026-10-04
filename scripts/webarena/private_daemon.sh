#!/usr/bin/env bash
# A second, private docker daemon for the WebArena sites, with its data on a large volume, leaving the machine's shared daemon
# (its /var/lib/docker data root and every other user's container) untouched.
#
#   sudo scripts/webarena/private_daemon.sh start     # bridge docker-wa (172.30.0.1/16) + dockerd on its own socket
#   sudo scripts/webarena/private_daemon.sh stop
#   scripts/webarena/private_daemon.sh env            # prints the `export DOCKER_HOST=...` line for this daemon
#
# Then every docker command for WebArena goes through this daemon without sudo (the socket belongs to the docker group):
#   export DOCKER_HOST=unix://$WA_DOCKER_ROOT/docker.sock      # WA_DOCKER_ROOT: the daemon's data root (default /raid/docker-wa)
#   scripts/webarena/host_sites.sh pull && scripts/webarena/host_sites.sh up
#
# Why a second daemon: /var/lib/docker holds 936 GB, so moving the shared daemon's data root to /raid would copy
# 750 GB more (hours of downtime for everyone) and leave /raid nearly full; the WebArena images need ~200 GB.
# Separation: own data-root, exec-root, pidfile, socket, config file and bridge; iptables management is off so the
# two daemons never touch the same chains (published ports still work through docker-proxy, containers just have no
# outbound NAT, which the sites do not need). dockerd spawns its own containerd under the exec root.
set -euo pipefail
ROOT=${WA_DOCKER_ROOT:-/raid/docker-wa}
SOCK="$ROOT/docker.sock"
BRIDGE=docker-wa
CONFIG="$ROOT/daemon.json"

case "${1:-}" in
  start)
    [ "$(id -u)" = 0 ] || { echo "run with sudo"; exit 1; }
    mkdir -p "$ROOT/data" "$ROOT/exec"
    cat > "$CONFIG" <<EOF
{
  "data-root": "$ROOT/data",
  "exec-root": "$ROOT/exec",
  "pidfile": "$ROOT/dockerd.pid",
  "hosts": ["unix://$SOCK"],
  "group": "docker",
  "bridge": "$BRIDGE",
  "iptables": false,
  "ip6tables": false,
  "ip-masq": false,
  "storage-driver": "overlay2",
  "log-level": "warn"
}
EOF
    if ! ip link show "$BRIDGE" >/dev/null 2>&1; then
      ip link add name "$BRIDGE" type bridge
      ip addr add 172.30.0.1/16 dev "$BRIDGE"
    fi
    ip link set "$BRIDGE" up
    sysctl -q -w net.ipv4.ip_forward=1
    if [ -f "$ROOT/dockerd.pid" ] && kill -0 "$(cat "$ROOT/dockerd.pid")" 2>/dev/null; then
      echo "private dockerd already running (pid $(cat "$ROOT/dockerd.pid"))"
    else
      nohup dockerd --config-file "$CONFIG" > "$ROOT/dockerd.log" 2>&1 &
      echo "private dockerd started (pid $!), log $ROOT/dockerd.log"
      for _ in $(seq 1 30); do [ -S "$SOCK" ] && break; sleep 1; done
    fi
    chgrp docker "$SOCK" 2>/dev/null || true; chmod 660 "$SOCK"
    DOCKER_HOST="unix://$SOCK" docker info --format 'root dir {{.DockerRootDir}}, server {{.ServerVersion}}' ;;
  stop)
    [ "$(id -u)" = 0 ] || { echo "run with sudo"; exit 1; }
    if [ -f "$ROOT/dockerd.pid" ]; then kill "$(cat "$ROOT/dockerd.pid")" 2>/dev/null || true; fi
    echo "stopped (bridge $BRIDGE left in place)" ;;
  env) echo "export DOCKER_HOST=unix://$SOCK" ;;
  *) sed -n 2,11p "$0"; exit 1 ;;
esac
