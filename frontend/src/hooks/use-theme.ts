import { useEffect, useState } from "react";

type Theme = "light" | "dark" | "system";

function getSystemTheme(): "light" | "dark" {
  return window.matchMedia("(prefers-color-scheme: dark)").matches
    ? "dark"
    : "light";
}

function resolveTheme(theme: Theme): "light" | "dark" {
  return theme === "system" ? getSystemTheme() : theme;
}

function applyTheme(theme: Theme) {
  const resolved = resolveTheme(theme);
  const el = document.documentElement;

  // Enable transition, apply class, then remove transition flag
  el.classList.add("transitioning");
  el.classList.toggle("dark", resolved === "dark");
  requestAnimationFrame(() => {
    setTimeout(() => el.classList.remove("transitioning"), 350);
  });
}

export function useTheme() {
  const [theme, setThemeRaw] = useState<Theme>(() => {
    return (localStorage.getItem("theme") as Theme | null) ?? "system";
  });

  const setTheme = (next: Theme) => {
    setThemeRaw(next);
    localStorage.setItem("theme", next);
    applyTheme(next);
  };

  const toggleTheme = () => {
    const resolved = resolveTheme(theme);
    setTheme(resolved === "dark" ? "light" : "dark");
  };

  // Sync on mount + listen for system preference changes
  useEffect(() => {
    applyTheme(theme);

    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const handler = () => {
      if (theme === "system") applyTheme("system");
    };
    mq.addEventListener("change", handler);
    return () => mq.removeEventListener("change", handler);
  }, [theme]);

  return {
    theme,
    resolvedTheme: resolveTheme(theme),
    setTheme,
    toggleTheme,
  };
}
