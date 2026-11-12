import { expect, test } from "@playwright/test";

test.describe("AXIOM landing page", () => {
  test.beforeEach(async ({ page }) => {
    await page.route(
      /^https:\/\/(?:fonts\.googleapis\.com|fonts\.gstatic\.com)\//,
      (route) => route.abort(),
    );
    await page.goto("/website/");
  });

  test("takes the install CTA to the install section", async ({ page }) => {
    const cliCtas = page.locator('a[href="#install"]');

    expect(await cliCtas.count()).toBeGreaterThan(0);
    await expect(cliCtas.first()).toHaveAttribute("href", "#install");

    await cliCtas.first().click();

    await expect(page).toHaveURL(/\/website\/#install$/);
    await expect(page.locator("#install")).toBeVisible();
    await expect(page.locator("#install")).toContainText("Install AXIOM");
  });

  test("switches the installer to the visitor's platform", async ({ page }) => {
    const unixPanel = page.locator('[data-platform-panel="unix"]');
    const windowsPanel = page.locator('[data-platform-panel="windows"]');
    const unixButton = page.getByRole("button", { name: "macOS / Linux" });
    const windowsButton = page.getByRole("button", { name: "Windows" });
    const detectedPlatform = await page.evaluate(() =>
      /win/i.test(navigator.userAgentData?.platform || navigator.platform || "")
        ? "windows"
        : "unix",
    );

    await expect(page.locator(`[data-platform-panel="${detectedPlatform}"]`)).toBeVisible();
    await expect(page.locator(`[data-platform="${detectedPlatform}"]`)).toHaveAttribute(
      "aria-pressed",
      "true",
    );

    await unixButton.click();

    await expect(unixPanel).toBeVisible();
    await expect(windowsPanel).toBeHidden();
    await expect(unixButton).toHaveAttribute("aria-pressed", "true");

    await windowsButton.click();

    await expect(windowsPanel).toBeVisible();
    await expect(unixPanel).toBeHidden();
    await expect(windowsButton).toHaveAttribute("aria-pressed", "true");
    await unixButton.click();
    await expect(unixPanel).toBeVisible();
    await expect(windowsPanel).toBeHidden();
  });

  test("shows the startup art and useful commands without captions or dividers", async ({ page }) => {
    await expect(page.locator(".cli-art")).toBeVisible();
    await expect(page.locator(".cli-command-list")).toContainText("axiom model inspect ./models/my-model");
    await expect(page.locator(".cli-command-list")).toContainText("axiom dataset validate ./data/train.jsonl");
    await expect(page.locator(".cli-command-list")).toContainText("axiom system info");
    await expect(page.locator(".cli-command-list span, .hero-cli figcaption")).toHaveCount(0);
    await expect(page.locator(".daemon-preview, .platform-note, .install-optional")).toHaveCount(0);

    const dividers = await page.locator(
      ".hero, .workflow-steps, .workflow-steps li, .cli-startup, .cli-command-list > div, .command-block, .site-footer",
    ).evaluateAll((elements) => elements.map((element) => {
      const style = getComputedStyle(element);
      return [style.borderTopWidth, style.borderRightWidth, style.borderBottomWidth, style.borderLeftWidth]
        .some((width) => Number.parseFloat(width) > 0);
    }));
    expect(dividers).toEqual(dividers.map(() => false));
  });

  test("keeps GitHub and CLI destinations explicit and network-independent", async ({ page }) => {
    const githubLinks = page.locator(
      'a[href="https://github.com/manit6752025/AXIOM-AI"]',
    );
    const cliCtas = page.locator('a[href="#install"]');

    expect(await githubLinks.count()).toBeGreaterThan(0);
    expect(await cliCtas.count()).toBeGreaterThan(0);

    const destinationChecks = [
      [githubLinks, "https://github.com/manit6752025/AXIOM-AI"],
      [cliCtas, "#install"],
    ] as const;

    for (const [links, destination] of destinationChecks) {
      const destinations = await links.evaluateAll((elements) =>
        elements.map((element) => element.getAttribute("href")),
      );
      expect(new Set(destinations)).toEqual(new Set([destination]));
    }

    await expect(githubLinks.first()).toHaveAttribute("target", "_blank");
    await expect(page.locator("#unix-command")).toHaveText(
      "curl -fsSL https://raw.githubusercontent.com/manit6752025/AXIOM-AI/main/installers/install.sh | bash",
    );
    await expect(page.locator("#windows-command")).toHaveText(
      "irm https://raw.githubusercontent.com/manit6752025/AXIOM-AI/main/installers/install.ps1 | iex",
    );
    await expect(page.getByRole("link", { name: "Check it out on GitHub" })).toBeVisible();
    await expect(page.locator("#install")).toContainText("run");
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

  test("keeps the landing page focused on the real CLI workflow", async ({ page }) => {
    await expect(page.locator("#capabilities")).toHaveCount(0);
    await expect(page.locator(".eyebrow, .image-label, .scroll-cue, figcaption")).toHaveCount(0);
    const favicon = page.locator('link[rel="icon"][sizes="any"]');
    const faviconHref = await favicon.getAttribute("href");
    expect(faviconHref).toBe("./assets/axiom-mark.svg");
    await expect(page.locator("header .brand-mark")).toHaveAttribute(
      "src",
      faviconHref ?? "",
    );
    await expect(page.locator("body")).not.toContainText("THE POINT");
    await expect(page.locator("#hero-title")).toContainText("Know what fits");
    await expect(page.locator("#hero-title")).toContainText("before you train.");
    await expect(page.locator(".hero-lead")).toContainText("Your files stay on your computer.");

    const sectionBackgrounds = await page.locator("body > main > section, .site-footer").evaluateAll((elements) =>
      elements.map((element) => getComputedStyle(element).backgroundColor),
    );
    expect(new Set(sectionBackgrounds).size).toBe(1);
  });
});
