import { access, cp, mkdir, readdir, rm } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const source = join(root, "dist");
const target = join(root, "axiom", "webui");

await access(join(source, "index.html")).catch(() => {
  throw new Error("dist/index.html is missing; run the Vite build first.");
});

await rm(target, { recursive: true, force: true });
await mkdir(target, { recursive: true });

for (const entry of await readdir(source)) {
  await cp(join(source, entry), join(target, entry), { recursive: true });
}

console.log(`Bundled Control Center copied to ${target}`);
