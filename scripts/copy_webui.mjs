import { randomUUID } from "node:crypto";
import { access, cp, mkdir, readdir, rename, rm } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const source = join(root, "dist");
const parent = join(root, "axiom");
const target = join(parent, "webui");
const suffix = `${process.pid}-${randomUUID()}`;
const staging = join(parent, `.webui-staging-${suffix}`);
const backup = join(parent, `.webui-previous-${suffix}`);

await access(join(source, "index.html")).catch(() => {
  throw new Error("dist/index.html is missing; run the Vite build first.");
});

await rm(staging, { recursive: true, force: true });
await mkdir(staging, { recursive: true });

let previousMoved = false;
let activated = false;
try {
  for (const entry of await readdir(source)) {
    await cp(join(source, entry), join(staging, entry), { recursive: true });
  }
  await access(join(staging, "index.html"));

  try {
    await rename(target, backup);
    previousMoved = true;
  } catch (error) {
    if (error && typeof error === "object" && "code" in error && error.code !== "ENOENT") {
      throw error;
    }
  }

  await rename(staging, target);
  activated = true;
  if (previousMoved) {
    await rm(backup, { recursive: true, force: true }).catch((error) => {
      console.warn(`Previous Control Center backup retained at ${backup}: ${error.message}`);
    });
  }
  console.log(`Bundled Control Center atomically activated at ${target}`);
} catch (error) {
  if (previousMoved && !activated) {
    await rename(backup, target).catch((restoreError) => {
      throw new Error(`Control Center activation failed and the previous bundle could not be restored: ${restoreError.message}`, { cause: error });
    });
  }
  throw error;
} finally {
  await rm(staging, { recursive: true, force: true });
}
