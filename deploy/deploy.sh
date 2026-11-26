#!/usr/bin/env bash
# Triển khai SpamGuard:
#   Trình duyệt ─HTTPS─► <project>.pages.dev (Pages Function) ─HTTP─► Nginx 58.187.192.54 ─LAN─► Docker 192.168.1.5
#
# Dùng:  deploy/deploy.sh <bước>
#   prepare   tạo deploy/.env (bí mật ngẫu nhiên) + registry đã lọc cho image
#   sync      đồng bộ mã nguồn lên server (jump host)
#   docker    build & chạy container trên Docker daemon 192.168.1.5 (qua server)
#   pages     tạo/deploy Cloudflare Pages project + cấu hình biến (cần `npx wrangler login` trước)
#   nginx     cài site Nginx trên server (cần sudo) – chạy sau `pages` để biết domain công khai
#   check     kiểm tra end-to-end
#   all       chạy lần lượt tất cả
#
# Biến môi trường tuỳ chọn:
#   SSH_CMD        lệnh SSH tới server (mặc định: ssh mnv@58.187.192.54)
#   RSYNC_RSH      lệnh SSH cho rsync (mặc định = SSH_CMD)
#   SUDO_PASSWORD  nếu đặt, dùng cho sudo -S trên server; không đặt thì sudo hỏi trực tiếp (ssh -t)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEPLOY="$ROOT/deploy"
ENV_FILE="$DEPLOY/.env"
SSH_CMD="${SSH_CMD:-ssh mnv@58.187.192.54}"
RSYNC_RSH="${RSYNC_RSH:-$SSH_CMD}"
REMOTE_DIR="spamguard-deploy"
DOCKER_HOST_LAN="tcp://192.168.1.5:2375"
UPSTREAM="192.168.1.5:8010"
ORIGIN_HOST="spamguard.58-187-192-54.sslip.io"
PAGES_PROJECT="${PAGES_PROJECT:-spamguard}"
REGISTRY_VERSIONS="1 2 3 4"   # chỉ đóng gói 4 model gốc của thí nghiệm CLO1

log() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
remote() { $SSH_CMD "$@"; }
env_get() { grep -E "^$1=" "$ENV_FILE" | cut -d= -f2-; }
rand() { LC_ALL=C tr -dc 'A-Za-z0-9' </dev/urandom | head -c "$1"; }

step_prepare() {
  log "Chuẩn bị bí mật và registry"
  if [[ ! -f "$ENV_FILE" ]]; then
    umask 077
    cat >"$ENV_FILE" <<EOF
ADMIN_TOKEN=$(rand 40)
POSTGRES_PASSWORD=$(rand 32)
GRAFANA_ADMIN_PASSWORD=$(rand 20)
ORIGIN_SECRET=$(rand 48)
API_PORT=8010
GRAFANA_PORT=3010
MLFLOW_PORT=5010
PROMETHEUS_PORT=9095
EOF
    echo "Đã tạo $ENV_FILE (quyền 600, không commit)"
  fi
  rm -rf "$DEPLOY/.build" && mkdir -p "$DEPLOY/.build/registry"
  for v in $REGISTRY_VERSIONS; do
    rsync -a --exclude hf_tokenizer "$ROOT/models/registry/v$v" "$DEPLOY/.build/registry/"
  done
  # registry.json chỉ giữ các version đã đóng gói
  python3 - "$ROOT/models/registry/registry.json" "$DEPLOY/.build/registry/registry.json" $REGISTRY_VERSIONS <<'PY'
import json, sys
src, dst, keep = sys.argv[1], sys.argv[2], {int(v) for v in sys.argv[3:]}
index = json.load(open(src))
index["versions"] = [v for v in index["versions"] if v["version"] in keep]
index["history"] = [h for h in index["history"] if h["to"] in keep and (h["from"] in keep or h["from"] is None)]
if index["aliases"].get("production") not in keep:
    raise SystemExit("Model production không nằm trong các version được đóng gói")
json.dump(index, open(dst, "w"), indent=2, ensure_ascii=False)
PY
  du -sh "$DEPLOY/.build/registry"
}

step_sync() {
  log "Đồng bộ mã nguồn lên server ~/$REMOTE_DIR"
  rsync -az --delete -e "$RSYNC_RSH" \
    --exclude .venv --exclude .git --exclude __pycache__ --exclude .pytest_cache --exclude .ruff_cache \
    --exclude models --exclude mlruns --exclude mlflow.db --exclude 'data/raw' --exclude 'data/*.db' \
    --exclude reports --exclude '*.docx' --exclude '*.xlsx' --exclude 'deploy/cloudflare-pages' \
    "$ROOT/" "mnv@58.187.192.54:$REMOTE_DIR/"
}

step_docker() {
  log "Build & chạy container trên $DOCKER_HOST_LAN"
  remote "cd $REMOTE_DIR && export DOCKER_HOST=$DOCKER_HOST_LAN && docker version --format 'Docker daemon: {{.Server.Version}} ({{.Server.Os}})' \
    && docker compose -f deploy/docker-compose.prod.yml --env-file deploy/.env up -d --build \
    && docker compose -f deploy/docker-compose.prod.yml --env-file deploy/.env ps"
}

step_pages() {
  log "Cloudflare Pages: project $PAGES_PROJECT"
  cd "$DEPLOY/cloudflare-pages"
  npx --yes wrangler@4 pages project list 2>/dev/null | grep -q " $PAGES_PROJECT " \
    || npx --yes wrangler@4 pages project create "$PAGES_PROJECT" --production-branch main
  npx --yes wrangler@4 pages deploy public --project-name "$PAGES_PROJECT" --branch main --commit-dirty=true
  # Biến cấu hình: ORIGIN_URL công khai, ORIGIN_SECRET là secret
  printf '%s' "http://$ORIGIN_HOST" | npx --yes wrangler@4 pages secret put ORIGIN_URL --project-name "$PAGES_PROJECT"
  env_get ORIGIN_SECRET | tr -d '\n' | npx --yes wrangler@4 pages secret put ORIGIN_SECRET --project-name "$PAGES_PROJECT"
  # Secret chỉ có hiệu lực ở lần deploy sau -> deploy lại
  npx --yes wrangler@4 pages deploy public --project-name "$PAGES_PROJECT" --branch main --commit-dirty=true
  cd "$ROOT"
}

public_host() {
  # Domain thực tế (Cloudflare có thể thêm hậu tố nếu tên đã bị dùng)
  local sub
  sub=$(cd "$DEPLOY/cloudflare-pages" && npx --yes wrangler@4 pages project list 2>/dev/null \
        | grep -Eo "[a-z0-9-]+\.pages\.dev" | grep -E "^$PAGES_PROJECT(-[a-z0-9]+)?\.pages\.dev$" | head -1)
  echo "${PUBLIC_HOST:-${sub:-$PAGES_PROJECT.pages.dev}}"
}

step_nginx() {
  local host conf
  host="$(public_host)"
  log "Cài site Nginx $ORIGIN_HOST → $UPSTREAM (Host công khai: $host)"
  conf="$(sed -e "s|__ORIGIN_HOST__|$ORIGIN_HOST|g" -e "s|__UPSTREAM__|$UPSTREAM|g" \
              -e "s|__PUBLIC_HOST__|$host|g" -e "s|__ORIGIN_SECRET__|$(env_get ORIGIN_SECRET)|g" \
              "$DEPLOY/nginx/spamguard.conf.template")"
  printf '%s\n' "$conf" | remote "umask 077; cat > ~/$REMOTE_DIR/spamguard.nginx.conf"
  # sudo đổi HOME thành /root -> dùng đường dẫn tuyệt đối của user deploy
  local remote_home staged install
  remote_home="$(remote 'printf %s "$HOME"')"
  staged="$remote_home/$REMOTE_DIR/spamguard.nginx.conf"
  install="install -m 600 $staged /etc/nginx/sites-available/$ORIGIN_HOST \
    && ln -sf /etc/nginx/sites-available/$ORIGIN_HOST /etc/nginx/sites-enabled/$ORIGIN_HOST \
    && nginx -t && systemctl reload nginx && rm -f $staged"
  if [[ -n "${SUDO_PASSWORD:-}" ]]; then
    printf '%s\n' "$SUDO_PASSWORD" | remote "sudo -S -p '' bash -c '$install'"
  else
    $SSH_CMD -t "sudo bash -c '$install'"
  fi
}

step_check() {
  local host
  host="$(public_host)"
  log "Kiểm tra"
  echo -n "LAN  (server → $UPSTREAM/ready): "; remote "curl -s --max-time 5 http://$UPSTREAM/ready"; echo
  echo -n "Origin không có header bí mật (mong đợi 403): "
  curl -s -o /dev/null -w '%{http_code}\n' "http://$ORIGIN_HOST/ready"
  echo -n "Công khai https://$host/ready: "; curl -s --max-time 15 "https://$host/ready"; echo
}

case "${1:-}" in
  prepare) step_prepare ;;
  sync) step_sync ;;
  docker) step_docker ;;
  pages) step_pages ;;
  nginx) step_nginx ;;
  check) step_check ;;
  all) step_prepare; step_sync; step_docker; step_pages; step_nginx; step_check ;;
  *) sed -n '2,20p' "$0"; exit 1 ;;
esac
