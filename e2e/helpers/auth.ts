import { webcrypto } from "node:crypto";
import { expect, type Page } from "@playwright/test";

const E2E_USERNAME = "axiom-e2e-admin";
const E2E_PASSWORD = "AxiomTest123";

function arrayBuffer(bytes: Uint8Array): ArrayBuffer {
  return bytes.slice().buffer as ArrayBuffer;
}

async function makeTestAdminRecord() {
  const salt = webcrypto.getRandomValues(new Uint8Array(16));
  const iv = webcrypto.getRandomValues(new Uint8Array(12));

  const passwordKey = await webcrypto.subtle.importKey(
    "raw",
    arrayBuffer(new TextEncoder().encode(E2E_PASSWORD)),
    "PBKDF2",
    false,
    ["deriveKey"],
  );

  const key = await webcrypto.subtle.deriveKey(
    {
      name: "PBKDF2",
      salt: arrayBuffer(salt),
      iterations: 310000,
      hash: "SHA-256",
    },
    passwordKey,
    { name: "AES-GCM", length: 256 },
    false,
    ["encrypt"],
  );

  const ciphertext = await webcrypto.subtle.encrypt(
    { name: "AES-GCM", iv: arrayBuffer(iv) },
    key,
    arrayBuffer(new TextEncoder().encode("AXIOM-ADMIN-VERIFIER-V1")),
  );

  return {
    version: 1,
    username: E2E_USERNAME,
    salt: Buffer.from(salt).toString("base64"),
    iv: Buffer.from(iv).toString("base64"),
    ciphertext: Buffer.from(ciphertext).toString("base64"),
    updatedAt: Date.now(),
  };
}

export async function ensureAuthenticated(page: Page): Promise<void> {
  const bootstrapFailures: string[] = [];
  page.on("pageerror", (error) => {
    bootstrapFailures.push(error.message);
    console.error("[AXIOM browser error]", error.message);
  });
  page.on("console", (message) => {
    if (message.type() === "error") {
      bootstrapFailures.push(message.text());
      console.error("[AXIOM console error]", message.text());
    }
  });
  page.on("requestfailed", (request) => {
    if (/\.(?:js|tsx?)(?:[?#]|$)/i.test(request.url())) {
      bootstrapFailures.push(`${request.url()}: ${request.failure()?.errorText ?? "request failed"}`);
    }
  });

  const response = await page.goto("/");
  if (response?.status() !== 200) {
    throw new Error(
      `AXIOM dev server returned HTTP ${response?.status() ?? "no response"} for /`,
    );
  }

  // Isolated Playwright context: seed only this test's administrator.
  const adminRecord = await makeTestAdminRecord();
  await page.evaluate((record) => {
    localStorage.clear();
    sessionStorage.clear();
    localStorage.setItem("axiom-booted", "1");
    localStorage.setItem("axiom-setup", "1");
    localStorage.setItem("axiom-admin-v2", JSON.stringify(record));
  }, adminRecord);

  await page.reload();

  const username = page.locator("#login-username");
  try {
    await expect(username).toBeVisible({ timeout: 10_000 });
  } catch {
    const state = await page.evaluate(() => ({
      url: location.href,
      title: document.title,
      bodyText: document.body.innerText.slice(0, 1000),
      rootHtml: document.querySelector("#root")?.innerHTML.slice(0, 2000) ?? null,
      entryScripts: Array.from(document.scripts).map((script) => script.src).filter(Boolean),
    })).catch(() => null);

    throw new Error(
      `AXIOM login form did not render after seeding test admin. State: ${JSON.stringify(state)}; browser errors: ${JSON.stringify(bootstrapFailures.slice(-12))}`,
    );
  }

  await username.fill(E2E_USERNAME);
  await page.locator("#login-password").fill(E2E_PASSWORD);

  const remember = page.getByRole("checkbox", {
    name: /Remember me on this browser/i,
  });
  await expect(remember).toBeVisible();
  await remember.check();

  await page.getByRole("button", { name: /^Sign in$/i }).click();
  await expect(page.getByLabel("Primary navigation")).toBeVisible({
    timeout: 15_000,
  });
}
