import { expect, test } from "@playwright/test";

test.describe("AXIOM landing page", () => {
  test.beforeEach(async ({ page }) => {
    await page.route(
      /^https:\/\/(?:fonts\.googleapis\.com|fonts\.gstatic\.com)\//,
      (route) => route.abort(),
    );
    await page.goto("/website/");
  });

  test("shows the hero and sends the main CTA to downloads", async ({ page }) => {
    await expect(page.locator(".home-hero h1")).toHaveAttribute("aria-label", "Plan local AI work before it runs.");
    await expect(page.locator(".home-hero")).toContainText("local-first, open source toolkit");

    const primary = page.locator(".home-hero .pro-button.primary");
    await expect(primary).toHaveText("Get AXIOM");
    await expect(primary).toHaveAttribute("href", "./downloads.html");
    await expect(page.locator(".home-hero .pro-button.secondary").first()).toHaveAttribute(
      "href",
      "./docs.html",
    );
  });

  test("shows the install command in both install blocks", async ({ page }) => {
    const command = "python -m pip install --upgrade --pre axiom-all";

    await expect(page.locator("#install")).toHaveText(command);
    await expect(page.locator("#install2")).toHaveText(command);
    await expect(page.locator('button[data-copy-target="install"]')).toBeVisible();
    await expect(page.locator('button[data-copy-target="install2"]')).toBeVisible();
  });

  test("shows the starter commands in the terminal demo", async ({ page }) => {
    const list = await page.locator("#cmd").getAttribute("data-list");

    expect(list).toContain("axiom doctor");
    expect(list).toContain("axiom system info");
    expect(list).toContain("axiom dataset validate ./data/train.jsonl");
    expect(list).toContain("axiom engine status");
  });

  test("links the main navigation pages", async ({ page }) => {
    const nav = page.locator("#pro-nav");
    const pages = [
      "platform",
      "control-center",
      "engine",
      "integrations",
      "workflows",
      "cli",
      "downloads",
      "docs",
      "roadmap",
      "security",
      "community",
    ];

    for (const name of pages) {
      await expect(nav.locator(`a[href="./${name}.html"]`)).toHaveCount(1);
    }
  });

  test("keeps the GitHub link explicit and uses the local favicon", async ({ page }) => {
    const github = page.locator('.pro-footer a[href="https://github.com/manit6752025/AXIOM-AI"]');

    await expect(github).toHaveCount(1);
    await expect(github).toHaveAttribute("target", "_blank");
    await expect(github).toHaveAttribute("rel", /noopener/);

    const favicon = page.locator('link[rel="icon"]');
    await expect(favicon).toHaveAttribute("href", "./assets/axiom-mark.svg");
    await expect(page.locator("header .brand-mark")).toHaveAttribute("src", "./assets/axiom-mark.svg");
  });

  test("does not render gradient styling or fake dashboard metrics", async ({ page }) => {
    const visibleCopy = await page.locator("body").innerText();
    expect(visibleCopy).not.toMatch(/\b\d+(?:\.\d+)?%\b/);
    expect(visibleCopy).not.toMatch(
      /\b(?:system health|inference activity|model registry|ready|synced|live)\b/i,
    );

    await expect(page.locator("[data-count], .metric-card, .gradient-icon")).toHaveCount(0);

    const gradientElements = await page.evaluate(() =>
      Array.from(document.querySelectorAll("body *"))
        .filter((element) => getComputedStyle(element).backgroundImage.includes("gradient"))
        .map((element) => element.tagName.toLowerCase()),
    );
    expect(gradientElements).toEqual([]);
  });
});
