import { spawn } from "node:child_process";

const python = process.env.AXIOM_PYTHON || (process.platform === "win32" ? "python" : "python3");
const apiUrl = "http://127.0.0.1:33855/api/health";
const children = [];

function start(command, args, name) {
  const child = spawn(command, args, {
    stdio: "inherit",
    env: process.env,
  });
  child.on("error", (error) => {
    console.error(`[${name}] ${error.message}`);
    process.exitCode = 1;
  });
  child.on("exit", (code, signal) => {
    if (signal && !process.exitCode) {
      process.exitCode = 1;
    } else if (code && !process.exitCode) {
      process.exitCode = code;
    }
  });
  children.push(child);
  return child;
}

async function waitForApi() {
  const deadline = Date.now() + 30_000;
  while (Date.now() < deadline) {
    try {
      const response = await fetch(apiUrl);
      if (response.ok) return;
    } catch {
      // The API process may need a few seconds to import dependencies.
    }
    if (children[0]?.exitCode !== null) {
      throw new Error("AXIOM API exited before becoming ready");
    }
    await new Promise((resolve) => setTimeout(resolve, 250));
  }
  throw new Error("AXIOM API did not become ready within 30 seconds");
}

function stopAll() {
  for (const child of children) {
    if (!child.killed) child.kill("SIGTERM");
  }
}

process.on("SIGINT", stopAll);
process.on("SIGTERM", stopAll);

try {
  start(
    python,
    ["-m", "uvicorn", "axiom.api.server:create_app", "--factory", "--host", "127.0.0.1", "--port", "33855"],
    "api",
  );
  await waitForApi();
  console.log("[api] ready at http://127.0.0.1:33855");
  start("npm", ["run", "dev"], "gui");
} catch (error) {
  console.error(`[gui] ${error instanceof Error ? error.message : String(error)}`);
  stopAll();
  process.exitCode = 1;
}
