# AXIOM frontend compiler and lint toolchain

AXIOM runs the native TypeScript 7.0.2 compiler while keeping the compatible
TypeScript 6.0.3 compiler API for typescript-eslint 8.73.0.

The root npm lock deliberately pins two package aliases:

- `@typescript/native` is the official `typescript@7.0.2` package and is
  invoked by an explicit path in the production build:
  `node node_modules/@typescript/native/bin/tsc -b`.
- `typescript` aliases `@typescript/typescript6@6.0.2`. That package
  exposes the TypeScript 6 API (6.0.3) through `@typescript/old`, which
  satisfies the current `typescript-eslint@8.73.0` peer range.

Do not replace the explicit compiler path with a bare `tsc` invocation:
both the native and compatibility packages provide compiler executables.
The prebuild/prelint verifier checks both exact compiler/API versions and the
typescript-eslint pin before a build or lint run.

Install with `npm ci`; do not hand-edit package-lock.json after dependency
updates. When upgrading TypeScript or typescript-eslint, update this
compatibility contract and re-run the complete lint, build and Playwright
quality matrix.
