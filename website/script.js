(() => {
  const siteHeader = document.querySelector(".site-header");
  const menuToggle = document.querySelector(".menu-toggle");
  const siteNav = document.querySelector(".site-nav");

  const closeMenu = () => {
    if (!siteHeader || !menuToggle) return;

    siteHeader.classList.remove("menu-open");
    menuToggle.setAttribute("aria-expanded", "false");
  };

  if (menuToggle && siteHeader) {
    menuToggle.addEventListener("click", () => {
      const isOpen = siteHeader.classList.toggle("menu-open");
      menuToggle.setAttribute("aria-expanded", String(isOpen));
    });

    document.addEventListener("click", (event) => {
      if (siteHeader.classList.contains("menu-open") && !siteHeader.contains(event.target)) {
        closeMenu();
      }
    });

    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape") closeMenu();
    });
  }

  siteNav?.addEventListener("click", (event) => {
    if (event.target instanceof Element && event.target.closest('a[href^="#"]')) {
      closeMenu();
    }
  });

  const platformButtons = document.querySelectorAll("[data-platform]");
  const platformPanels = document.querySelectorAll("[data-platform-panel]");
  const detectedPlatform = /win/i.test(
    navigator.userAgentData?.platform || navigator.platform || "",
  )
    ? "windows"
    : "unix";

  const setPlatform = (platform) => {
    platformButtons.forEach((button) => {
      const active = button.dataset.platform === platform;
      button.classList.toggle("is-active", active);
      button.setAttribute("aria-pressed", String(active));
    });

    platformPanels.forEach((panel) => {
      const active = panel.dataset.platformPanel === platform;
      panel.hidden = !active;
      panel.classList.toggle("is-active", active);
    });
  };

  platformButtons.forEach((button) => {
    button.addEventListener("click", () => setPlatform(button.dataset.platform));
  });
  setPlatform(detectedPlatform);

  const copyText = async (text) => {
    if (navigator.clipboard && window.isSecureContext) {
      try {
        await navigator.clipboard.writeText(text);
        return;
      } catch {
        // Fall through when the browser denies clipboard access.
      }
    }

    const helper = document.createElement("textarea");
    helper.value = text;
    helper.setAttribute("readonly", "");
    helper.style.position = "fixed";
    helper.style.opacity = "0";
    document.body.appendChild(helper);
    helper.select();
    const copied = document.execCommand("copy");
    helper.remove();

    if (!copied) throw new Error("Clipboard copy was unavailable");
  };

  document.querySelectorAll("[data-copy]").forEach((button) => {
    button.addEventListener("click", async () => {
      const source = document.getElementById(button.dataset.copy);
      const value = source?.textContent.trim();
      if (!value) return;

      const statusId = button.getAttribute("aria-describedby");
      const status = statusId ? document.getElementById(statusId) : null;
      const defaultStatus = status?.dataset.defaultText || "";
      if (status && !status.dataset.defaultText) {
        status.dataset.defaultText = status.textContent.trim();
      }

      try {
        await copyText(value);
        button.textContent = "Copied";
        if (status) status.textContent = "Copied to your clipboard.";

        window.setTimeout(() => {
          button.textContent = "Copy";
          if (status) status.textContent = defaultStatus;
        }, 2200);
      } catch {
        if (status) status.textContent = "Copy failed. Select the command manually.";
      }
    });
  });
})();
