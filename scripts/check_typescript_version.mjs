import { createRequire } from "node:module";
import { readFileSync } from "node:fs";

const require = createRequire(import.meta.url);
const packageJson = JSON.parse(readFileSync(new URL("../package.json", import.meta.url), "utf8"));
const devDependencies = packageJson.devDependencies ?? {};
const expectedNative = devDependencies["@typescript/native"];
const expectedApi = devDependencies.typescript;
if (expectedNative !== "npm:typescript@7.0.2" || expectedApi !== "npm:@typescript/typescript6@6.0.2") {
  console.error("AXIOM frontend requires the pinned TypeScript 7 compiler and TypeScript 6 ESLint API compatibility packages.");
  process.exit(1);
}

let apiVersion;
let nativeVersion;
try {
  apiVersion = require("typescript").version;
  nativeVersion = JSON.parse(
    readFileSync(new URL("../node_modules/@typescript/native/package.json", import.meta.url), "utf8"),
  ).version;
} catch {
  console.error("AXIOM frontend TypeScript toolchain is incomplete. Run npm ci in the repository root.");
  process.exit(1);
}
if (apiVersion !== "6.0.3" || nativeVersion !== "7.0.2") {
  console.error(
    "AXIOM frontend toolchain mismatch: expected TypeScript 6.0.3 API / 7.0.2 compiler, " +
    "but found API " + apiVersion + " / compiler " + nativeVersion + ". Run npm ci to restore the locked toolchain.",
  );
  process.exit(1);
}

const typescriptEslint = devDependencies["typescript-eslint"];
if (typescriptEslint !== "8.73.0") {
  console.error("AXIOM frontend requires the tested typescript-eslint 8.73.0 toolchain pin.");
  process.exit(1);
}
console.log(
  "AXIOM frontend toolchain verified: TypeScript " + nativeVersion +
  " compiler, TypeScript " + apiVersion + " ESLint API, typescript-eslint " + typescriptEslint + ".",
);
