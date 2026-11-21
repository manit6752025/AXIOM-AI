import { createRequire } from "node:module";
import { existsSync, readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";

const require = createRequire(import.meta.url);
let entry;
try {
  entry = require.resolve("lucide-react");
} catch {
  console.error("AXIOM frontend lucide-react is not installed. Run npm ci in the repository root.");
  process.exit(1);
}

const packageRoot = resolve(dirname(entry), "..", "..");
let packageJson;
try {
  packageJson = JSON.parse(readFileSync(resolve(packageRoot, "package.json"), "utf8"));
} catch {
  console.error("AXIOM frontend lucide-react package metadata is missing. Run npm ci in the repository root.");
  process.exit(1);
}

const declaration = packageJson.types ?? packageJson.typings ?? packageJson.exports?.["."]?.types;
if (
  typeof declaration !== "string" ||
  declaration.startsWith("/") ||
  declaration.split(/[\\/]/).includes("..") ||
  !existsSync(resolve(packageRoot, declaration))
) {
  console.error(
    "AXIOM frontend lucide-react TypeScript declarations are missing from node_modules. " +
    "The package lock already includes bundled types; run npm ci in the repository root to restore them.",
  );
  process.exit(1);
}

console.log("AXIOM frontend icon declarations verified: " + declaration + ".");