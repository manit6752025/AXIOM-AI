"""Static contract checks for reproducible AXIOM release and brand wiring."""

from __future__ import annotations

import re
import stat
import tomllib
from html.parser import HTMLParser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class _AssetParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[dict[str, str]] = []
        self.images: list[dict[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = {key: value or "" for key, value in attrs}
        if tag == "link":
            self.links.append(attributes)
        elif tag == "img":
            self.images.append(attributes)


def _runtime_version() -> str:
    source = (ROOT / "axiom/version.py").read_text(encoding="utf-8")
    match = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', source, re.MULTILINE)
    assert match, "axiom/version.py must define __version__"
    return match.group(1)


def test_project_metadata_runtime_and_release_workflow_share_one_version():
    metadata = tomllib.loads(
        (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    project_version = metadata["project"]["version"]
    runtime_version = _runtime_version()
    workflow = (ROOT / ".github/workflows/release-assets.yml").read_text(
        encoding="utf-8"
    )

    assert project_version == runtime_version
    assert f'default: "v{runtime_version}"' in workflow


def test_linux_installer_verifies_the_published_bundle_before_user_path_update():
    installer_path = ROOT / "installers/install.sh"
    installer = installer_path.read_text(encoding="utf-8")

    assert installer_path.stat().st_mode & stat.S_IXUSR
    assert 'set -euo pipefail' in installer
    assert 'REPO="manit6752025/AXIOM-AI"' in installer
    assert 'asset="AXIOM-${tag}-linux-x64.tar.gz"' in installer
    assert 'download_file "$download_base/SHA256SUMS"' in installer
    assert 'verify_checksum "$DOWNLOAD_DIR/$asset" "$asset"' in installer
    assert 'validate_archive "$DOWNLOAD_DIR/$asset"' in installer
    assert 'actual="$(sha256_file "$file")"' in installer
    assert '[[ "$actual" == "$expected" ]]' in installer
    assert '[[ -f "$extracted_dir/AXIOM/AXIOM" && -x "$extracted_dir/AXIOM/AXIOM" ]]' in installer
    assert 'release_dir="$INSTALL_DIR/releases/$tag"' in installer
    assert 'link_path "$INSTALL_DIR/current/AXIOM" "$BIN_DIR/axiom"' in installer
    assert 'AXIOM_RELEASE_TAG' in installer
    assert 'AXIOM_INSTALL_DIR' in installer
    assert 'AXIOM_BIN_DIR' in installer


def test_release_workflow_publishes_and_checksums_every_linux_install_asset():
    workflow = (ROOT / ".github/workflows/release-assets.yml").read_text(
        encoding="utf-8"
    )
    linux_assets = (
        "AXIOM-${TAG}-linux-x64.deb",
        "AXIOM-${TAG}-linux-x64.AppImage",
        "AXIOM-${TAG}-linux-x64.sh",
        "AXIOM-${TAG}-linux-x64.elf",
        "AXIOM-${TAG}-linux-x64.tar.gz",
    )

    for asset in linux_assets:
        assert f'"release/{asset}"' in workflow

    assert 'tar -tzf "out/AXIOM-${TAG}-linux-x64.tar.gz" >"$bundle_manifest"' in workflow
    assert 'grep -q \'^AXIOM/AXIOM$\' "$bundle_manifest"' in workflow
    assert "files: release/*" in workflow
    assert 'sha256sum "${release_binaries[@]#./}" > SHA256SUMS' in workflow
    assert "No AXIOM release binaries found for checksumming." in workflow


def test_pypi_workflow_publishes_validated_distributions_on_version_tags():
    workflow = (ROOT / ".github/workflows/publish-pypi.yml").read_text(
        encoding="utf-8"
    )
    metadata = tomllib.loads(
        (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )

    assert metadata["project"]["name"] == "axiom-all"
    assert 'tags:\n      - "v*"' in workflow
    assert "bash scripts/build_webui.sh" in workflow
    assert "python -m build --sdist --wheel --outdir python-dist" in workflow
    assert "python -m twine check python-dist/*.whl python-dist/*.tar.gz" in workflow
    assert "Expected exactly one wheel and one source archive" in workflow
    assert "Verify downloaded artifacts contain only distributions" in workflow
    assert "python -m twine check dist/*" not in workflow
    assert "axiom/webui/index.html" in workflow
    assert "axiom version" in workflow
    assert "pypa/gh-action-pypi-publish@release/v1" in workflow
    assert "environment:\n      name: pypi" in workflow
    assert "secrets.PYPI_API_TOKEN" in workflow


def test_website_uses_the_saturn_mark_for_favicon_and_brand_logo():
    html = (ROOT / "website/index.html").read_text(encoding="utf-8")
    parser = _AssetParser()
    parser.feed(html)

    favicon_links = [
        link
        for link in parser.links
        if link.get("rel", "").lower() == "icon"
    ]
    assert len(favicon_links) == 1
    favicon = favicon_links[0]
    assert favicon["href"] == "./assets/axiom-mark.svg"
    assert favicon["type"] == "image/svg+xml"

    brand_images = [
        image
        for image in parser.images
        if "brand-mark" in image.get("class", "").split()
    ]
    assert len(brand_images) >= 2
    assert {image["src"] for image in brand_images} == {favicon["href"]}

    mark = (ROOT / "website/assets/axiom-mark.svg").read_text(encoding="utf-8")
    assert 'viewBox="0 0 128 128"' in mark
    assert "AXIOM Saturn mark" in mark
    assert "<ellipse" in mark
    assert "<path" in mark
    assert 'fill="#171310"' in mark
