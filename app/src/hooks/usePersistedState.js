import { useState, useEffect } from "react";

// All persisted keys live under a single namespace so they're easy to spot
// (and clear) in devtools → Application → Local Storage.
const PREFIX = "snakeai:";

export function readPersisted(key, fallback) {
  try {
    const raw = localStorage.getItem(PREFIX + key);
    return raw === null ? fallback : JSON.parse(raw);
  } catch {
    return fallback;
  }
}

export function writePersisted(key, value) {
  try {
    localStorage.setItem(PREFIX + key, JSON.stringify(value));
  } catch {
    /* quota exceeded or storage disabled — non-fatal */
  }
}

export function removePersisted(key) {
  try {
    localStorage.removeItem(PREFIX + key);
  } catch {
    /* non-fatal */
  }
}

// useState that survives a full page reload by mirroring its value to
// localStorage. The reload itself is external (Vite dev server / WebView
// memory pressure); this just makes it non-destructive.
export function usePersistedState(key, initial) {
  const [value, setValue] = useState(() =>
    readPersisted(key, typeof initial === "function" ? initial() : initial)
  );

  useEffect(() => {
    writePersisted(key, value);
  }, [key, value]);

  return [value, setValue];
}
