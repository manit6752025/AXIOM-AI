#!/usr/bin/env bash
set -euo pipefail

# AXIOM installer contract
#
#   AXIOM_RELEASE_TAG   Optional exact release tag, for example v0.3.0-beta.6.
#                       When unset, the newest published non-draft GitHub release is selected.
#   AXIOM_SOURCE_REF    Optional Git ref for the source fallback. It defaults
#                       to AXIOM_RELEASE_TAG when pinned, otherwise main.
#   AXIOM_INSTALL_DIR   User-scoped runtime/cache directory. Defaults to
#                       ~/.local/share/axiom.
#   AXIOM_BIN_DIR       Directory for the axiom symlink. Defaults to
#                       ~/.local/bin.
#
# All path overrides must be absolute paths. The x86_64 Linux path downloads
# and verifies the onedir bundle because it starts faster on repeated runs.

REPO="NetCore-Technologies/AXIOM-AI"
USER_HOME="${HOME:-}"
INSTALL_DIR="${AXIOM_INSTALL_DIR:-${USER_HOME}/.local/share/axiom}"
BIN_DIR="${AXIOM_BIN_DIR:-${USER_HOME}/.local/bin}"
REQUESTED_TAG="${AXIOM_RELEASE_TAG:-}"
SOURCE_REF_OVERRIDE="${AXIOM_SOURCE_REF:-}"
DOWNLOAD_DIR=""
AXIOM_BIN=""

die() {
  echo "AXIOM installer error: $*" >&2
  exit 1
}

show_help() {
  cat <<'EOF'
AXIOM installer

Usage:
  curl -fsSL https://raw.githubusercontent.com/NetCore-Technologies/AXIOM-AI/main/installers/install.sh | bash

Optional environment overrides:
  AXIOM_RELEASE_TAG=v0.3.0-beta.6  Install one exact GitHub release.
  AXIOM_SOURCE_REF=main             Git ref for the source fallback.
  AXIOM_INSTALL_DIR=/path/to/data   User-scoped runtime and release storage.
  AXIOM_BIN_DIR=/path/to/bin        Directory receiving the axiom symlink.

The default Linux x86_64 install is the fast onedir release bundle. Every
downloaded bundle is checked against SHA256SUMS and its archive layout is
validated before activation.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  show_help
  exit 0
fi
[[ "$#" -eq 0 ]] || die "unknown argument '$1'; use --help for usage"

[[ -n "$USER_HOME" ]] || die "HOME is not set; set HOME before running the installer"

validate_path() {
  local name="$1"
  local path="$2"
  [[ "$path" == /* ]] || die "$name must be an absolute path (received '$path')"
  [[ "$path" != "/" ]] || die "$name cannot be the filesystem root"
}

validate_path AXIOM_INSTALL_DIR "$INSTALL_DIR"
validate_path AXIOM_BIN_DIR "$BIN_DIR"

valid_release_tag() {
  [[ "$1" =~ ^v[0-9]+\.[0-9]+\.[0-9]+([.-][0-9A-Za-z.-]+)?$ ]]
}

if [[ -n "$REQUESTED_TAG" ]] && ! valid_release_tag "$REQUESTED_TAG"; then
  die "AXIOM_RELEASE_TAG must look like v0.3.0-beta.6 (received '$REQUESTED_TAG')"
fi

cleanup_download() {
  if [[ -n "$DOWNLOAD_DIR" && -d "$DOWNLOAD_DIR" ]]; then
    rm -rf -- "$DOWNLOAD_DIR"
  fi
}

trap cleanup_download EXIT

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | awk '{print $1}'
  else
    die "sha256sum or shasum is required to verify the release"
  fi
}

download_file() {
  local url="$1"
  local destination="$2"
  if ! curl --fail --location --silent --show-error \
    --retry 3 --retry-delay 1 --connect-timeout 10 --max-time 180 \
    "$url" -o "$destination"; then
    die "download failed: $url; check your network connection and release tag"
  fi
  [[ -s "$destination" ]] || die "download was empty: $url"
}

latest_release_tag() {
  local response="$DOWNLOAD_DIR/releases.json"
  download_file "https://api.github.com/repos/$REPO/releases?per_page=20" "$response"
  awk '
    /"tag_name"[[:space:]]*:/ {
      line = $0
      sub(/^.*"tag_name"[[:space:]]*:[[:space:]]*"/, "", line)
      sub(/".*$/, "", line)
      print line
      exit
    }
  ' "$response"
}

link_path() {
  local target="$1"
  local link="$2"
  if [[ -e "$link" && ! -L "$link" ]]; then
    die "refusing to replace existing non-symlink: $link"
  fi
  ln -sfn -- "$target" "$link"
}

validate_archive() {
  local archive="$1"
  local manifest="$DOWNLOAD_DIR/archive.manifest"

  if ! tar -tzf "$archive" >"$manifest"; then
    die "could not read the downloaded AXIOM archive; it may be incomplete"
  fi

  if ! awk '
    {
      entry = $0
      if (entry == "AXIOM/AXIOM") found = 1
      if (entry !~ /^AXIOM(\/|$)/ || entry ~ /(^|\/)\.\.(\/|$)/ || entry ~ /^\//) invalid = 1
    }
    END { exit !(found && !invalid) }
  ' "$manifest"; then
    die "refusing to extract an AXIOM archive with an unexpected layout"
  fi
}

verify_checksum() {
  local file="$1"
  local asset="$2"
  local sums="$3"
  local expected actual

  expected="$(awk -v asset="$asset" '
    NF >= 2 {
      name = $2
      sub(/^\*/, "", name)
      if (name == asset) { print $1; exit }
    }
  ' "$sums")"
  expected="$(printf '%s' "$expected" | tr '[:upper:]' '[:lower:]')"
  [[ "$expected" =~ ^[[:xdigit:]]{64}$ ]] || die "SHA256SUMS does not list a valid checksum for $asset"

  actual="$(sha256_file "$file")"
  actual="$(printf '%s' "$actual" | tr '[:upper:]' '[:lower:]')"
  [[ "$actual" == "$expected" ]] || die "checksum verification failed for $asset"
}

install_linux_bundle() {
  local tag="$1"
  local asset="AXIOM-${tag}-linux-x64.tar.gz"
  local download_base="https://github.com/$REPO/releases/download/$tag"
  local release_dir="$INSTALL_DIR/releases/$tag"
  local extracted_dir="$DOWNLOAD_DIR/extracted"

  echo "Downloading the efficient Linux bundle ($tag)..."
  download_file "$download_base/$asset" "$DOWNLOAD_DIR/$asset"
  download_file "$download_base/SHA256SUMS" "$DOWNLOAD_DIR/SHA256SUMS"
  verify_checksum "$DOWNLOAD_DIR/$asset" "$asset" "$DOWNLOAD_DIR/SHA256SUMS"
  validate_archive "$DOWNLOAD_DIR/$asset"

  mkdir -p "$extracted_dir" "$INSTALL_DIR/releases"
  if ! tar -xzf "$DOWNLOAD_DIR/$asset" --no-same-owner -C "$extracted_dir"; then
    die "could not extract the verified AXIOM archive"
  fi
  [[ -f "$extracted_dir/AXIOM/AXIOM" && -x "$extracted_dir/AXIOM/AXIOM" ]] || \
    die "the verified Linux bundle is missing its executable"

  if [[ -e "$release_dir" || -L "$release_dir" ]]; then
    [[ -d "$release_dir" && ! -L "$release_dir" ]] || die "refusing to replace unsafe release path: $release_dir"
    rm -rf -- "$release_dir"
  fi
  mv "$extracted_dir/AXIOM" "$release_dir"
  link_path "$release_dir" "$INSTALL_DIR/current"
  link_path "$INSTALL_DIR/current/AXIOM" "$BIN_DIR/axiom"
  AXIOM_BIN="$BIN_DIR/axiom"
}

install_from_source() {
  local source_ref="$1"
  echo "Installing AXIOM from source ref $source_ref for this platform..."
  command -v python3 >/dev/null 2>&1 || die "Python 3.11+ is required for the source fallback"
  command -v git >/dev/null 2>&1 || die "git is required for the source fallback"
  python3 -m venv "$INSTALL_DIR/venv"
  "$INSTALL_DIR/venv/bin/python" -m pip install --upgrade pip >/dev/null || \
    die "could not bootstrap pip in $INSTALL_DIR/venv"
  "$INSTALL_DIR/venv/bin/python" -m pip install --upgrade "git+https://github.com/$REPO.git@$source_ref" || \
    die "could not install AXIOM from source ref $source_ref"
  link_path "$INSTALL_DIR/venv/bin/axiom" "$BIN_DIR/axiom"
  AXIOM_BIN="$INSTALL_DIR/venv/bin/axiom"
}

echo "AXIOM terminal installer"
echo
mkdir -p "$INSTALL_DIR" "$BIN_DIR"
DOWNLOAD_DIR="$(mktemp -d "$INSTALL_DIR/.download.XXXXXX")"

case "$(uname -s)" in
  Darwin)
    SOURCE_REF="${SOURCE_REF_OVERRIDE:-${REQUESTED_TAG:-main}}"
    install_from_source "$SOURCE_REF"
    ;;
  Linux)
    ARCH="$(uname -m)"
    if [[ "$ARCH" == "x86_64" ]]; then
      command -v curl >/dev/null 2>&1 || die "curl is required to download the Linux bundle"
      command -v tar >/dev/null 2>&1 || die "tar is required to install the Linux bundle"
      TAG="${REQUESTED_TAG:-$(latest_release_tag)}"
      [[ -n "$TAG" ]] || die "could not determine an AXIOM release; set AXIOM_RELEASE_TAG explicitly"
      valid_release_tag "$TAG" || die "GitHub returned an invalid AXIOM release tag: $TAG"
      install_linux_bundle "$TAG"
    else
      SOURCE_REF="${SOURCE_REF_OVERRIDE:-${REQUESTED_TAG:-main}}"
      install_from_source "$SOURCE_REF"
    fi
    ;;
  *)
    die "unsupported operating system. Use the PowerShell installer on Windows"
    ;;
esac

[[ -x "$AXIOM_BIN" ]] || die "installation completed without an executable at $AXIOM_BIN"

echo
echo "AXIOM installed at $AXIOM_BIN"
if [[ ":$PATH:" != *":$BIN_DIR:"* ]]; then
  echo "Add it to your PATH with:"
  echo "  export PATH=\"$BIN_DIR:\$PATH\""
fi
echo
"$AXIOM_BIN" --no-banner version
