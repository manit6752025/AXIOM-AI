#!/usr/bin/env node
import { spawnSync } from "node:child_process";
const args = process.argv.slice(2);
const candidates = process.platform === "win32" ? [["py", ["-m", "axiom.cli.main", ...args]], ["python", ["-m", "axiom.cli.main", ...args]]] : [["python3", ["-m", "axiom.cli.main", ...args]], ["python", ["-m", "axiom.cli.main", ...args]]];
for (const [command, commandArgs] of candidates) { const result = spawnSync(command, commandArgs, { stdio: "inherit" }); if (result.error?.code === "ENOENT") continue; process.exit(result.status ?? 1); }
console.error("Install the Python package first: python -m pip install axiom-all"); process.exit(127);
