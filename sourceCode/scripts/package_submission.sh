#!/usr/bin/env bash
# Đóng gói bài nộp với đúng cấu trúc repo:
#   HTTM_Nhom15_SpamGuard/
#   ├── docs/         tài liệu báo cáo
#   ├── sourceCode/   mã nguồn (chỉ file cần thiết)
#   └── README.md     giới thiệu + hướng dẫn chạy
#
# Bỏ qua mọi thứ tự sinh lại được khi build (theo .gitignore):
#   .venv/ (thư viện)        -> make setup
#   data/raw/ (dữ liệu gốc)  -> make data
#   models/ (model đã train) -> make train  (hoặc make train-fast)
#   deploy/.build/, *.db, mlflow.db, mlruns/, cache
# và bí mật triển khai sourceCode/deploy/.env (không bao giờ được nộp).
#
# Dùng (trong sourceCode/): make package  -> sourceCode/dist/HTTM_Nhom15_SpamGuard.zip
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
NAME="${1:-HTTM_Nhom15_SpamGuard.zip}"
OUT_DIR="$REPO/sourceCode/dist"
PREFIX="HTTM_Nhom15_SpamGuard"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

cd "$REPO"
mkdir -p "$OUT_DIR"
rm -f "$OUT_DIR/$NAME"

# File được git theo dõi + file mới chưa commit nhưng không bị .gitignore loại
git ls-files --cached --others --exclude-standard -z \
  | grep -zv -E '^(sourceCode/dist/|sourceCode/deploy/\.env$)' \
  | while IFS= read -r -d '' file; do
      [[ -f "$file" ]] || continue
      mkdir -p "$STAGE/$PREFIX/$(dirname "$file")"
      cp -p "$file" "$STAGE/$PREFIX/$file"
    done

# Chốt chặn cuối: tuyệt đối không để lọt bí mật / thư mục nặng
for forbidden in sourceCode/deploy/.env sourceCode/.venv sourceCode/models/registry sourceCode/data/raw sourceCode/mlflow.db; do
  if [[ -e "$STAGE/$PREFIX/$forbidden" ]]; then
    echo "LỖI: $forbidden lọt vào gói nộp" >&2
    exit 1
  fi
done

(cd "$STAGE" && zip -qr -9 "$OUT_DIR/$NAME" "$PREFIX" -x '*.DS_Store')
echo "Đã tạo $OUT_DIR/$NAME ($(du -h "$OUT_DIR/$NAME" | cut -f1), $(unzip -Z1 "$OUT_DIR/$NAME" | grep -vc '/$') file)"
