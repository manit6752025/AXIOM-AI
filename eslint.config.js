import js from "@eslint/js";
import { defineConfig } from "eslint/config";
import tseslint from "typescript-eslint";

export default defineConfig([
  {
    ignores: [
      "node_modules/**",
      "dist/**",
      "coverage/**",
      "axiom/webui/**",
    ],
  },
  {
    files: ["**/*.{js,mjs,cjs,ts,tsx}"],
    extends: [
      js.configs.recommended,
      tseslint.configs.recommended,
    ],
    rules: {
      "no-undef": "off",
    },
  },
]);
