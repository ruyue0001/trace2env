#!/usr/bin/env bash
# Host WebArena's four sites (shopping, shopping_admin, gitlab, forum) under the AgentWorldBench hostnames for
# scripts/collect_webarena.py. Needs docker without sudo and ~200 GB in the daemon's data root for the four images;
# on the shared machine that daemon is the private one from private_daemon.sh (export DOCKER_HOST first).
#
#   scripts/webarena/host_sites.sh pull        # docker pull the four images from Docker Hub (webarenaimages/*, ~140 GB)
#   scripts/webarena/host_sites.sh load DIR    # or docker load the four image tars found in DIR
#   scripts/webarena/host_sites.sh up          # start the containers and the :80 reverse proxy, set the sites' base URLs
#   scripts/webarena/host_sites.sh check       # HTTP status of each site through the proxy
#   scripts/webarena/host_sites.sh reset       # recreate the site containers from the images (fresh database state)
#   scripts/webarena/host_sites.sh down        # stop and remove everything (images stay loaded)
#
# Images: Docker Hub hosts them as webarenaimages/shopping_final_0712, webarenaimages/shopping_admin_final_0719,
# webarenaimages/gitlab-populated-final and webarenaimages/postmill-populated-exposed-withimg (2025-03 uploads; the
# WebArena README's archive.org items are dark, its CMU mirror answers 403 and its Google Drive files exceed their
# download quota, checked 2026-09-23). `pull` tags them with the names `start_sites` uses. Browsers reach the sites
# through Chromium host-resolver rules that scripts/collect_webarena.py passes to Playwright MCP (--resolve-to, default
# 127.0.0.1); nothing touches /etc/hosts.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROXY=webarena-proxy

# The sites share one user-defined network whose subnet must be free on the host: the private daemon's default bridge
# (172.30.0.0/16) collided with a compose network of the shared daemon, which silently broke host-to-container traffic.
NET=${WA_NET:-wa-net}
SUBNET=${WA_SUBNET:-10.213.0.0/16}

ensure_network() {
  docker network inspect "$NET" >/dev/null 2>&1 || docker network create --subnet "$SUBNET" "$NET" >/dev/null
}

start_sites() {
  ensure_network
  docker run --name shopping --network "$NET" -p 7770:80 -d shopping_final_0712
  docker run --name shopping_admin --network "$NET" -p 7780:80 -d shopping_admin_final_0719
  # GitLab keeps its Prometheus metrics as memory-mapped files under /dev/shm; docker's default 64 MB fills within
  # minutes and every request then fails with "IOError (unmapped file)" (HTTP 500). 1 GB is what the machine's
  # shared daemon uses as its default.
  docker run --name gitlab --network "$NET" --shm-size 1g -d -p 8023:8023 gitlab-populated-final-port8023 /opt/gitlab/embedded/bin/runsvdir-start
  docker run --name forum --network "$NET" -p 9999:80 -d postmill-populated-exposed-withimg
}

configure_sites() {
  echo "waiting 90 s for Magento and 5 min for GitLab to boot ..."
  sleep 90
  docker exec shopping /var/www/magento2/bin/magento setup:store-config:set --base-url="http://magento-store.example.com"
  docker exec shopping mysql -u magentouser -pMyPassword magentodb -e 'UPDATE core_config_data SET value="http://magento-store.example.com/" WHERE path = "web/secure/base_url";'
  docker exec shopping /var/www/magento2/bin/magento cache:flush
  docker exec shopping_admin /var/www/magento2/bin/magento setup:store-config:set --base-url="http://magento-admin.example.com"
  docker exec shopping_admin mysql -u magentouser -pMyPassword magentodb -e 'UPDATE core_config_data SET value="http://magento-admin.example.com/" WHERE path = "web/secure/base_url";'
  docker exec shopping_admin php /var/www/magento2/bin/magento config:set admin/security/password_is_forced 0
  docker exec shopping_admin php /var/www/magento2/bin/magento config:set admin/security/password_lifetime 0
  docker exec shopping_admin /var/www/magento2/bin/magento cache:flush
  sleep 210
  docker exec gitlab update-permissions
  docker exec gitlab sed -i "s|^external_url.*|external_url 'http://gitlab.example.com'|" /etc/gitlab/gitlab.rb
  docker exec gitlab bash -c "grep -q \"^nginx\\['listen_port'\\]\" /etc/gitlab/gitlab.rb || echo \"nginx['listen_port'] = 8023\" >> /etc/gitlab/gitlab.rb"
  # Omnibus sizes puma from the CPU count; on a 128-core host the workers exhaust PostgreSQL's 200 connection slots
  # ("FATAL: remaining connection slots are reserved", GitLab's static 500 page in front of Rails). Size for one
  # browser instead; appended unconditionally because the stock gitlab.rb carries these keys as comments.
  docker exec gitlab bash -c "printf '%s\n' \"puma['worker_processes'] = 4\" \"puma['min_threads'] = 4\" \"puma['max_threads'] = 4\" \"sidekiq['max_concurrency'] = 10\" \"postgresql['max_connections'] = 400\" >> /etc/gitlab/gitlab.rb"
  docker exec gitlab gitlab-ctl reconfigure
  # reconfigure restarts nginx and workhorse but leaves puma running with memory maps of files that update-permissions
  # and the reconfigure rewrote; every logged-in request then fails with "IOError (unmapped file)" (HTTP 500) until
  # puma is restarted. Restart everything, then give it ~90 s.
  docker exec gitlab gitlab-ctl restart
  sleep 90
}

start_proxy() {
  docker rm -f "$PROXY" >/dev/null 2>&1 || true
  docker run -d --name "$PROXY" --network host -v "$HERE/nginx.conf:/etc/nginx/conf.d/default.conf:ro" nginx:alpine
}

check() {
  for host in gitlab.example.com magento-store.example.com magento-admin.example.com forum.example.com; do
    printf '%-28s ' "$host"
    curl -s -o /dev/null -w '%{http_code}\n' --max-time 30 --resolve "$host:80:127.0.0.1" "http://$host/" || echo "unreachable"
  done
}

case "${1:-}" in
  pull)
    for pair in shopping_final_0712:shopping_final_0712 shopping_admin_final_0719:shopping_admin_final_0719 \
                gitlab-populated-final:gitlab-populated-final-port8023 postmill-populated-exposed-withimg:postmill-populated-exposed-withimg; do
      echo "pulling webarenaimages/${pair%%:*}"; docker pull "webarenaimages/${pair%%:*}:latest"
      docker tag "webarenaimages/${pair%%:*}:latest" "${pair##*:}"
    done
    docker images | grep -E "shopping_final_0712|shopping_admin_final_0719|gitlab-populated-final-port8023|postmill-populated-exposed-withimg" ;;
  load)
    for tar in shopping_final_0712.tar shopping_admin_final_0719.tar gitlab-populated-final-port8023.tar postmill-populated-exposed-withimg.tar; do
      echo "loading $tar"; docker load --input "$2/$tar"
    done ;;
  up) start_sites; start_proxy; configure_sites; check ;;
  check) check ;;
  reset)
    docker rm -f shopping shopping_admin gitlab forum >/dev/null 2>&1 || true
    start_sites; configure_sites; check ;;
  down) docker rm -f shopping shopping_admin gitlab forum "$PROXY" >/dev/null 2>&1 || true; echo "stopped" ;;
  *) sed -n 2,15p "$0"; exit 1 ;;
esac
