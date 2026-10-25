# AXIOM container images

AXIOM publishes images to GitHub Container Registry (GHCR) from verified commits. The workflow runs the release smoke checks, full Python tests, ESLint, the production UI build, and Playwright browser tests before publishing. It builds the core image for Linux AMD64 and ARM64; the inference, training, and combined full-extras variants are Linux AMD64.

## Published packages

| Image | Purpose | Platforms |
| --- | --- | --- |
| `ghcr.io/manit6752025/axiom-ai` | CLI, Control Center, daemon and workspace API; no optional PyTorch extras | Linux AMD64, ARM64 |
| `ghcr.io/manit6752025/axiom-ai-inference` | Core app plus the `native-inference` Python extras | Linux AMD64 |
| `ghcr.io/manit6752025/axiom-ai-training` | Core app plus the `training` Python extras | Linux AMD64 |
| `ghcr.io/manit6752025/axiom-ai-full` | Core app plus both `native-inference` and `training` extras | Linux AMD64 |

Each package receives `sha-<commit>` tags. Verified development pushes also update `latest-development`, nightly integration pushes update `nightly`, main pushes update `latest`, and version tags are published as-is. The source repository label links the packages back to this repository.

## Run the core image

```bash
docker pull ghcr.io/manit6752025/axiom-ai:latest
docker run --rm -p 127.0.0.1:8765:8765 ghcr.io/manit6752025/axiom-ai:latest
```

Open [http://127.0.0.1:8765/ui/](http://127.0.0.1:8765/ui/) for the Control Center, or check [http://127.0.0.1:8765/health](http://127.0.0.1:8765/health). Compose configuration is provided in `compose.yaml`.

## Choose an image for the branch you're testing

```bash
# Development build
docker pull ghcr.io/manit6752025/axiom-ai:latest-development

# Native local inference
docker pull ghcr.io/manit6752025/axiom-ai-inference:latest-development

# Training dependencies
docker pull ghcr.io/manit6752025/axiom-ai-training:latest-development

# Full inference + fine-tuning dependencies
docker pull ghcr.io/manit6752025/axiom-ai-full:latest-development
```

For a tagged release, replace the tag with the exact release tag, such as `v0.3.0-beta.6`. Model weights are not bundled in any image. GPU device access and drivers remain host/runtime-specific; this does not guarantee CUDA or ROCm support in every container environment.

## Security notes

The image runs as unprivileged UID/GID `10001:10001`. The daemon intentionally enables a non-loopback bind inside the container, so publish its host port only to loopback unless you have separately reviewed and secured network access. Do not expose the full workspace API to the public internet. The included Compose file maps the host port to `127.0.0.1`, drops Linux capabilities, enables `no-new-privileges`, and uses a temporary filesystem for `/tmp`.

When bind-mounting a writable host workspace on Linux, make its directory writable by UID 10001 (for example, `sudo chown -R 10001:10001 ./workspace`). Be aware that files created by the container will have that UID on the host. Only mount directories that the container is meant to access.
