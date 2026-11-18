#!/usr/bin/env bash
# Deploy only committed frontend/static files and the project's Nginx server.
set -euo pipefail

SSH_HOST="${1:-Aliyun}"
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "Refusing to deploy uncommitted tracked changes" >&2
  exit 1
fi
RELEASE="$(git rev-parse HEAD)"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
git archive --format=tar.gz --output="$STAGE/release.tar.gz" HEAD web deploy/aliyun/search.shwstone.cn.conf
printf '%s\n' "$RELEASE" >"$STAGE/REVISION"

ssh "$SSH_HOST" "mkdir -p /opt/nlp-search-engine/releases/$RELEASE /opt/nlp-search-engine/operations"
scp -q "$STAGE/release.tar.gz" "$SSH_HOST:/opt/nlp-search-engine/releases/$RELEASE/release.tar.gz"
scp -q "$STAGE/REVISION" "$SSH_HOST:/opt/nlp-search-engine/releases/$RELEASE/REVISION"
ssh "$SSH_HOST" "set -euo pipefail
  cd /opt/nlp-search-engine/releases/$RELEASE
  tar -xzf release.tar.gz
  rm release.tar.gz
  test -s web/index.html
  test -s web/app.js
  test -s web/style.css
  cp /etc/nginx/sites-enabled/search.shwstone.cn.conf /opt/nlp-search-engine/operations/search-https.before-$RELEASE.conf
  install -m 0644 deploy/aliyun/search.shwstone.cn.conf /etc/nginx/sites-enabled/search.shwstone.cn.conf
  if ! nginx -t; then
    cp /opt/nlp-search-engine/operations/search-https.before-$RELEASE.conf /etc/nginx/sites-enabled/search.shwstone.cn.conf
    nginx -t
    exit 1
  fi
  ln -s /opt/nlp-search-engine/releases/$RELEASE /opt/nlp-search-engine/current.next
  mv -Tf /opt/nlp-search-engine/current.next /opt/nlp-search-engine/current
  systemctl reload nginx
  curl -fsS --max-time 5 http://127.0.0.1:18081/gateway/health >/dev/null
  curl -fsS --max-time 10 https://search.shwstone.cn/ | grep -q '知海'
"
echo "Deployed $RELEASE to https://search.shwstone.cn via $SSH_HOST"
