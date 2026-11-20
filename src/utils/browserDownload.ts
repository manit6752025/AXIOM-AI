const MAX_DOWNLOAD_FILENAME_LENGTH = 180;
const FALLBACK_DOWNLOAD_FILENAME = "axiom-export.bin";

/** Keep only a safe leaf name before assigning the browser download attribute. */
export function sanitizeDownloadFilename(requestedFilename: string): string {
  const leaf = requestedFilename.replaceAll("\\", "/").split("/").filter(Boolean).at(-1) ?? "";
  let safe = leaf
    .normalize("NFKC")
    .replace(/[\u0000-\u001f\u007f]/g, "")
    .replace(/[<>:"|?*]/g, "_")
    .replace(/^[. ]+|[. ]+$/g, "")
    .trim()
    .slice(0, MAX_DOWNLOAD_FILENAME_LENGTH);

  if (!safe) safe = FALLBACK_DOWNLOAD_FILENAME;
  const deviceName = safe.split(".", 1)[0].toUpperCase();
  if (/^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])$/.test(deviceName)) safe = `_${safe}`;
  return safe;
}

/**
 * Initiate a browser-managed download. The helper does not write into the
 * workspace or upload the Blob. Keep the object URL alive briefly so browsers
 * can start the download before it is revoked.
 */
export function downloadBlob(blob: Blob, requestedFilename: string): string {
  if (!(blob instanceof Blob)) throw new TypeError("A browser Blob is required for download.");

  const filename = sanitizeDownloadFilename(requestedFilename);
  const objectUrl = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = objectUrl;
  anchor.download = filename;
  anchor.hidden = true;

  try {
    document.body.appendChild(anchor);
    anchor.click();
  } catch (error) {
    URL.revokeObjectURL(objectUrl);
    throw error;
  } finally {
    anchor.remove();
  }

  window.setTimeout(() => URL.revokeObjectURL(objectUrl), 1_000);
  return filename;
}
