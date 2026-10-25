#!/usr/bin/env bash
set -euo pipefail

# Build the Linux release from one source tree.  The one-file executable is
# useful for a quick portable download; the onedir bundle is the efficient
# path for repeated CLI and daemon launches because it does not unpack itself
# into a temporary directory on every run.

TAG="${1:?usage: build_linux_release.sh <tag> [output-dir] [appimagetool] }"
OUT_DIR="${2:-out}"
APPIMAGE_TOOL="${3:-${APPIMAGE_TOOL:-}}"
ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ ! "$TAG" =~ ^v[0-9]+\.[0-9]+\.[0-9]+([.-][0-9A-Za-z.-]+)?$ ]]; then
  echo "release tag must look like v0.3.0-beta.6" >&2
  exit 2
fi

mkdir -p "$OUT_DIR"
OUT_DIR="$(CDPATH= cd -- "$OUT_DIR" && pwd)"
if [[ "$OUT_DIR" == "/" || "$OUT_DIR" == "$ROOT_DIR" ]]; then
  echo "refusing to use the repository root or filesystem root as output" >&2
  exit 2
fi

BUILD_DIR="$ROOT_DIR/build/linux-release"
ONEFILE_DIST="$BUILD_DIR/onefile-dist"
ONEDIR_DIST="$BUILD_DIR/onedir-dist"
PACKAGE_DIR="$BUILD_DIR/package"
APP_DIR="$BUILD_DIR/AppDir"

rm -rf "$BUILD_DIR" "$OUT_DIR"
mkdir -p "$ONEFILE_DIST" "$ONEDIR_DIST" "$OUT_DIR"

PYTHON="${PYTHON:-python}"
PYINSTALLER=("$PYTHON" -m PyInstaller --noconfirm --clean)

"${PYINSTALLER[@]}" \
  --onefile \
  --name AXIOM \
  --collect-data axiom \
  --distpath "$ONEFILE_DIST" \
  --workpath "$BUILD_DIR/onefile-work" \
  --specpath "$BUILD_DIR/onefile-spec" \
  "$ROOT_DIR/scripts/build_entry.py"

install -m 0755 "$ONEFILE_DIST/AXIOM" \
  "$OUT_DIR/AXIOM-${TAG}-linux-x64.elf"

"${PYINSTALLER[@]}" \
  --onedir \
  --name AXIOM \
  --collect-data axiom \
  --distpath "$ONEDIR_DIST" \
  --workpath "$BUILD_DIR/onedir-work" \
  --specpath "$BUILD_DIR/onedir-spec" \
  "$ROOT_DIR/scripts/build_entry.py"

PORTABLE_DIR="$ONEDIR_DIST/AXIOM"
if [[ ! -x "$PORTABLE_DIR/AXIOM" ]]; then
  echo "PyInstaller did not produce the expected onedir executable" >&2
  exit 1
fi

# Stable tar metadata keeps the downloadable bundle reproducible for a given
# source checkout when the runner supplies SOURCE_DATE_EPOCH.
SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH:-$(git -C "$ROOT_DIR" log -1 --format=%ct 2>/dev/null || date +%s)}"
tar \
  --sort=name \
  --mtime="@${SOURCE_DATE_EPOCH}" \
  --owner=0 \
  --group=0 \
  --numeric-owner \
  -C "$ONEDIR_DIST" \
  -czf "$OUT_DIR/AXIOM-${TAG}-linux-x64.tar.gz" \
  AXIOM

mkdir -p "$PACKAGE_DIR/DEBIAN" "$PACKAGE_DIR/usr/bin" "$PACKAGE_DIR/usr/lib/axiom-ai"
cp -a "$PORTABLE_DIR/." "$PACKAGE_DIR/usr/lib/axiom-ai/"

cat > "$PACKAGE_DIR/usr/bin/axiom" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
exec /usr/lib/axiom-ai/AXIOM "$@"
EOF
chmod 0755 "$PACKAGE_DIR/usr/bin/axiom"

DEBIAN_VERSION="${TAG#v}"
DEBIAN_VERSION="${DEBIAN_VERSION//-beta./~beta.}"
DEBIAN_VERSION="${DEBIAN_VERSION//-b./~b.}"
cat > "$PACKAGE_DIR/DEBIAN/control" <<EOF
Package: axiom-ai
Version: ${DEBIAN_VERSION}
Section: devel
Priority: optional
Architecture: amd64
Maintainer: NetCore Technologies
Description: AXIOM local AI CLI and daemon
 AXIOM inspects AI work, checks the local machine, and exposes a loopback daemon.
EOF

dpkg-deb --build --root-owner-group "$PACKAGE_DIR" \
  "$OUT_DIR/AXIOM-${TAG}-linux-x64.deb" >/dev/null

if [[ -n "$APPIMAGE_TOOL" ]]; then
  mkdir -p "$APP_DIR/usr/lib/axiom-ai"
  cp -a "$PORTABLE_DIR/." "$APP_DIR/usr/lib/axiom-ai/"

  cat > "$APP_DIR/AppRun" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
HERE="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$HERE/usr/lib/axiom-ai/AXIOM" "$@"
EOF
  chmod 0755 "$APP_DIR/AppRun"

  cat > "$APP_DIR/axiom.desktop" <<'EOF'
[Desktop Entry]
Type=Application
Name=AXIOM
Comment=Local AI CLI and daemon
Exec=axiom
Icon=axiom
Terminal=true
Categories=Development;
EOF

  cp "$ROOT_DIR/website/assets/axiom-mark.svg" "$APP_DIR/axiom.svg"
  ARCH=x86_64 "$APPIMAGE_TOOL" \
    "$APP_DIR" \
    "$OUT_DIR/AXIOM-${TAG}-linux-x64.AppImage"
else
  echo "AppImage tool not supplied; skipping AppImage build." >&2
fi

# This companion installer works when the tarball, checksum file, and script
# are downloaded together. It mirrors the repository install contract.
INSTALLER="$OUT_DIR/AXIOM-${TAG}-linux-x64.sh"
cat > "$INSTALLER" <<EOF
#!/usr/bin/env bash
set -euo pipefail

# Optional user-scoped overrides: AXIOM_INSTALL_DIR and AXIOM_BIN_DIR.
# Both must be absolute paths; the tarball and SHA256SUMS stay beside this script.

TAG="${TAG}"
SOURCE_DIR="\$(CDPATH= cd -- "\$(dirname -- "\${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="\${AXIOM_INSTALL_DIR:-\${HOME}/.local/share/axiom}"
TARGET_DIR="\${AXIOM_BIN_DIR:-\${HOME}/.local/bin}"
TARGET="\${TARGET_DIR}/axiom"
BUNDLE="\${SOURCE_DIR}/AXIOM-${TAG}-linux-x64.tar.gz"
RAW="\${SOURCE_DIR}/AXIOM-${TAG}-linux-x64.elf"
CHECKSUMS="\${SOURCE_DIR}/SHA256SUMS"
RELEASE_DIR="\${INSTALL_DIR}/releases"
RUNTIME_DIR="\${RELEASE_DIR}/\${TAG}"

die() {
  echo "AXIOM installer error: \$*" >&2
  exit 1
}

validate_path() {
  local name="\$1"
  local path="\$2"
  [[ "\$path" == /* ]] || die "\$name must be an absolute path"
  [[ "\$path" != "/" ]] || die "\$name cannot be the filesystem root"
}

[[ -n "\${HOME:-}" ]] || die "HOME is not set"
validate_path AXIOM_INSTALL_DIR "\$INSTALL_DIR"
validate_path AXIOM_BIN_DIR "\$TARGET_DIR"

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "\$1" | awk '{print \$1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "\$1" | awk '{print \$1}'
  else
    die "sha256sum or shasum is required to verify the release"
  fi
}

verify_checksum() {
  local file="\$1"
  local asset="\$2"
  local expected actual
  [[ -f "\$CHECKSUMS" ]] || die "SHA256SUMS is missing next to the release assets"
  expected="\$(awk -v asset="\$asset" '
    NF >= 2 {
      name = \$2
      sub(/^\*/, "", name)
      if (name == asset) { print \$1; exit }
    }
  ' "\$CHECKSUMS")"
  expected="\$(printf '%s' "\$expected" | tr '[:upper:]' '[:lower:]')"
  [[ "\$expected" =~ ^[[:xdigit:]]{64}$ ]] || die "SHA256SUMS has no valid checksum for \$asset"
  actual="\$(sha256_file "\$file")"
  actual="\$(printf '%s' "\$actual" | tr '[:upper:]' '[:lower:]')"
  [[ "\$actual" == "\$expected" ]] || die "checksum verification failed for \$asset"
}

link_path() {
  local target="\$1"
  local link="\$2"
  if [[ -e "\$link" && ! -L "\$link" ]]; then
    die "refusing to replace existing non-symlink: \$link"
  fi
  ln -sfn -- "\$target" "\$link"
}

validate_archive() {
  local archive="\$1"
  local manifest="\${TEMP_DIR}/archive.manifest"
  tar -tzf "\$archive" >"\$manifest" || die "could not read the AXIOM tarball"
  awk '
    {
      entry = \$0
      if (entry == "AXIOM/AXIOM") found = 1
      if (entry !~ /^AXIOM(\/|$)/ || entry ~ /(^|\/)\.\.(\/|$)/ || entry ~ /^\//) invalid = 1
    }
    END { exit !(found && !invalid) }
  ' "\$manifest" || die "refusing to extract an AXIOM archive with an unexpected layout"
}

mkdir -p "\${TARGET_DIR}" "\${RELEASE_DIR}" "\${INSTALL_DIR}"
TEMP_DIR="\$(mktemp -d "\${INSTALL_DIR}/.install.XXXXXX")"
cleanup() { rm -rf -- "\${TEMP_DIR}"; }
trap cleanup EXIT

if [[ -f "\${BUNDLE}" ]]; then
  verify_checksum "\${BUNDLE}" "AXIOM-${TAG}-linux-x64.tar.gz"
  validate_archive "\${BUNDLE}"
  tar -xzf "\${BUNDLE}" --no-same-owner -C "\${TEMP_DIR}" || die "could not extract the verified AXIOM tarball"
  [[ -f "\${TEMP_DIR}/AXIOM/AXIOM" && -x "\${TEMP_DIR}/AXIOM/AXIOM" ]] || die "AXIOM tarball is missing its executable"
  if [[ -e "\${RUNTIME_DIR}" || -L "\${RUNTIME_DIR}" ]]; then
    [[ -d "\${RUNTIME_DIR}" && ! -L "\${RUNTIME_DIR}" ]] || die "refusing to replace unsafe release path"
    rm -rf -- "\${RUNTIME_DIR}"
  fi
  mv "\${TEMP_DIR}/AXIOM" "\${RUNTIME_DIR}"
  link_path "\${RUNTIME_DIR}" "\${INSTALL_DIR}/current"
  link_path "\${INSTALL_DIR}/current/AXIOM" "\${TARGET}"
elif [[ -f "\${RAW}" && -x "\${RAW}" ]]; then
  verify_checksum "\${RAW}" "AXIOM-${TAG}-linux-x64.elf"
  mkdir -p "\${RUNTIME_DIR}"
  install -m 0755 "\${RAW}" "\${RUNTIME_DIR}/axiom"
  link_path "\${RUNTIME_DIR}" "\${INSTALL_DIR}/current"
  link_path "\${INSTALL_DIR}/current/axiom" "\${TARGET}"
else
  die "download the tarball or single-file executable beside this installer, then run it again"
fi

echo "AXIOM installed to \${TARGET}"
"\${TARGET}" --no-banner version
EOF
chmod 0755 "$INSTALLER"

# Keep a Linux-only checksum file beside the companion installer. The publish
# job later replaces it with checksums that also cover the Windows assets.
CHECKSUM_FILES=(
  "AXIOM-${TAG}-linux-x64.elf"
  "AXIOM-${TAG}-linux-x64.tar.gz"
  "AXIOM-${TAG}-linux-x64.deb"
  "AXIOM-${TAG}-linux-x64.sh"
)
if [[ -f "$OUT_DIR/AXIOM-${TAG}-linux-x64.AppImage" ]]; then
  CHECKSUM_FILES+=("AXIOM-${TAG}-linux-x64.AppImage")
fi
(
  cd "$OUT_DIR"
  sha256sum "${CHECKSUM_FILES[@]}"
) > "$OUT_DIR/SHA256SUMS"

echo "Built Linux assets in $OUT_DIR"
find "$OUT_DIR" -maxdepth 1 -type f -printf '%f\n' | sort
