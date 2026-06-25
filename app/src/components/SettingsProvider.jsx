import { createContext, useContext, useEffect } from "react";
import { usePersistedState } from "../hooks/usePersistedState";

/**
 * App-wide visual preferences (accent color, UI density, reduced motion).
 *
 * The whole stylesheet keys its primary accent off the `--green` family
 * (`--green` / `--green-soft` / `--green-glow`) plus a few helper vars added in
 * App.css (`--accent-hover`, `--accent-ring`, `--accent-shadow`). Changing the
 * accent here just re-points those CSS variables at runtime, so every primary
 * button, nav highlight, focus ring, and "running" indicator recolors at once
 * with no per-component wiring. Density toggles a root attribute the stylesheet
 * reads; reduced-motion sets a root attribute *and* is respected by the
 * `prefers-reduced-motion` media query for users who set it at the OS level.
 */

export const ACCENTS = [
  { id: "emerald", label: "Emerald", base: "#10b981", hover: "#34d399" },
  { id: "blue",    label: "Azure",   base: "#3b82f6", hover: "#60a5fa" },
  { id: "purple",  label: "Violet",  base: "#8b5cf6", hover: "#a78bfa" },
  { id: "cyan",    label: "Cyan",    base: "#06b6d4", hover: "#22d3ee" },
  { id: "orange",  label: "Amber",   base: "#f59e0b", hover: "#fbbf24" },
  { id: "rose",    label: "Rose",    base: "#f43f5e", hover: "#fb7185" },
];

const DEFAULT_ACCENT = "emerald";

function hexToRgb(hex) {
  const h = hex.replace("#", "");
  return [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16));
}
function rgba([r, g, b], a) {
  return `rgba(${r}, ${g}, ${b}, ${a})`;
}

function applyAccent(id) {
  const accent = ACCENTS.find((a) => a.id === id) || ACCENTS[0];
  const rgb = hexToRgb(accent.base);
  const root = document.documentElement.style;
  // Re-point the accent family the whole stylesheet already consumes…
  root.setProperty("--green", accent.base);
  root.setProperty("--green-soft", rgba(rgb, 0.12));
  root.setProperty("--green-glow", rgba(rgb, 0.4));
  root.setProperty("--border-focus", rgba(rgb, 0.5));
  // …plus the helper vars App.css now uses where the accent was hardcoded.
  root.setProperty("--accent-hover", accent.hover);
  root.setProperty("--accent-ring", rgba(rgb, 0.15));
  root.setProperty("--accent-shadow", rgba(rgb, 0.25));
  root.setProperty("--accent-shadow-strong", rgba(rgb, 0.35));
}

const SettingsContext = createContext(null);

export function useSettings() {
  const ctx = useContext(SettingsContext);
  if (!ctx) throw new Error("useSettings must be used within SettingsProvider");
  return ctx;
}

export function SettingsProvider({ children }) {
  const [accent, setAccent] = usePersistedState("pref:accent", DEFAULT_ACCENT);
  const [density, setDensity] = usePersistedState("pref:density", "comfortable");
  const [reducedMotion, setReducedMotion] = usePersistedState("pref:reducedMotion", false);

  useEffect(() => { applyAccent(accent); }, [accent]);

  useEffect(() => {
    document.documentElement.setAttribute("data-density", density);
  }, [density]);

  useEffect(() => {
    document.documentElement.toggleAttribute("data-reduce-motion", reducedMotion);
  }, [reducedMotion]);

  const value = {
    accent, setAccent,
    density, setDensity,
    reducedMotion, setReducedMotion,
    accents: ACCENTS,
  };

  return <SettingsContext.Provider value={value}>{children}</SettingsContext.Provider>;
}
