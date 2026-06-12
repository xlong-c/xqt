(function () {
  const root = document.documentElement;
  const themeKey = "xdl-doc-theme";
  const accentKey = "xdl-doc-accent";
  const allowedThemes = new Set(["system", "light", "dark", "sepia"]);
  const allowedAccents = new Set(["teal", "blue", "violet", "amber", "rose", "green"]);

  function readChoice(key, fallback, allowed) {
    const value = window.localStorage.getItem(key) || fallback;
    return allowed.has(value) ? value : fallback;
  }

  function applyTheme(value) {
    const theme = allowedThemes.has(value) ? value : "system";
    if (theme === "system") {
      root.setAttribute("data-theme", "system");
    } else {
      root.setAttribute("data-theme", theme);
    }
    window.localStorage.setItem(themeKey, theme);
    updatePressed("[data-theme-value]", theme);
  }

  function applyAccent(value) {
    const accent = allowedAccents.has(value) ? value : "teal";
    if (accent === "teal") {
      root.removeAttribute("data-accent");
    } else {
      root.setAttribute("data-accent", accent);
    }
    window.localStorage.setItem(accentKey, accent);
    updatePressed("[data-accent-value]", accent);
  }

  function updatePressed(selector, value) {
    document.querySelectorAll(selector).forEach((button) => {
      const key = button.getAttribute(selector.slice(1, -1));
      button.setAttribute("aria-pressed", String(key === value));
    });
  }

  function bindControls() {
    document.querySelectorAll("[data-theme-value]").forEach((button) => {
      button.addEventListener("click", () => applyTheme(button.dataset.themeValue || "system"));
    });

    document.querySelectorAll("[data-accent-value]").forEach((button) => {
      button.addEventListener("click", () => applyAccent(button.dataset.accentValue || "teal"));
    });

    updatePressed("[data-theme-value]", readChoice(themeKey, "system", allowedThemes));
    updatePressed("[data-accent-value]", readChoice(accentKey, "teal", allowedAccents));
  }

  applyTheme(readChoice(themeKey, "system", allowedThemes));
  applyAccent(readChoice(accentKey, "teal", allowedAccents));

  window.addEventListener("DOMContentLoaded", bindControls);

  window.XDLTheme = {
    setTheme: applyTheme,
    setAccent: applyAccent,
    getState: function () {
      return {
        theme: readChoice(themeKey, "system", allowedThemes),
        accent: readChoice(accentKey, "teal", allowedAccents),
      };
    },
  };
})();
