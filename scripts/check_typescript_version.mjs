import { createRequire } from "node:module";
import { readFileSync } from "node:fs";

const require = createRequire(import.meta.url);
const packageJson = JSON.parse(readFileSync(new URL("../package.json", import.meta.url), "utf8"));
const expected = packageJson.devDependencies?.typescript;
if (typeof expected !== "string" || !/^\d+\.\d+\.\d+$/.test(expected)) {
  console.error("AXIOM frontend requires an exact TypeScript devDependency pin.");
  process.exit(1);
}

let actual;
try {
  actual = require("typescript").version;
} catch (error) {
  console.error("AXIOM frontend TypeScript is not installed. Run npm ci in the repository root.");
  process.exit(1);
}
if (actual !== expected) {
  console.error(
    "AXIOM frontend toolchain mismatch: package.json pins TypeScript " + expected +
    ", but node_modules provides " + actual + ". Run npm ci to restore the locked toolchain.",
  );
  process.exit(1);
}

const typescriptEslint = packageJson.devDependencies?.["typescript-eslint"];
if (typescriptEslint !== "8.73.0") {
  console.error("AXIOM frontend requires the tested typescript-eslint 8.73.0 toolchain pin.");
  process.exit(1);
}
console.log("AXIOM frontend toolchain verified: TypeScript " + actual + ", typescript-eslint " + typescriptEslint + ".");
