import { useState, useRef, useEffect, useCallback } from "react";
import { invoke } from "@tauri-apps/api/core";

// Maps a backend process id to a human label, its sidebar panel, and an icon.
// Keep in sync with the ids passed to start_process across the panels.
const PROC_META = {
  "train":            { label: "Training",          panel: "train",     icon: "⚡" },
  "benchmark":        { label: "Benchmark",         panel: "benchmark", icon: "📊" },
  "init_model":       { label: "Initializing model", panel: "models",    icon: "✨" },
  "play-watch":       { label: "Watch AI",          panel: "play",      icon: "🤖" },
};

const metaFor = (id) => PROC_META[id] || { label: id, panel: null, icon: "›" };

/**
 * Always-available view of running Python subprocesses. The trigger pill lives in
 * the status bar; clicking it opens a fixed-positioned popover (fixed, not
 * absolute, so it can't be clipped by the scrolling main column) listing each
 * process with a jump-to-panel action and an individual stop, plus a Stop all.
 */
export default function ProcessTray({ running, onNavigate }) {
  const [open, setOpen] = useState(false);
  const [coords, setCoords] = useState(null);
  const triggerRef = useRef(null);
  const popRef = useRef(null);

  const count = running.length;

  const place = useCallback(() => {
    const el = triggerRef.current;
    if (!el) return;
    const r = el.getBoundingClientRect();
    setCoords({ top: r.bottom + 8, right: Math.max(8, window.innerWidth - r.right) });
  }, []);

  useEffect(() => {
    if (!open) return;
    place();
    const onDocClick = (e) => {
      if (popRef.current?.contains(e.target) || triggerRef.current?.contains(e.target)) return;
      setOpen(false);
    };
    const onKey = (e) => e.key === "Escape" && setOpen(false);
    document.addEventListener("mousedown", onDocClick);
    document.addEventListener("keydown", onKey);
    window.addEventListener("resize", place);
    return () => {
      document.removeEventListener("mousedown", onDocClick);
      document.removeEventListener("keydown", onKey);
      window.removeEventListener("resize", place);
    };
  }, [open, place]);

  // Close automatically once nothing is left to manage.
  useEffect(() => { if (count === 0) setOpen(false); }, [count]);

  const stop = (id) => invoke("stop_process", { id }).catch((e) => console.error(e));
  const stopAll = () => running.forEach(stop);

  const jump = (panel) => {
    if (panel) onNavigate?.(panel);
    setOpen(false);
  };

  if (count === 0) {
    return (
      <div className="tray-pill tray-pill--idle" title="No processes running">
        <span className="tray-dot tray-dot--idle" />
        Idle
      </div>
    );
  }

  return (
    <>
      <button
        ref={triggerRef}
        className="tray-pill tray-pill--live"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        aria-haspopup="dialog"
        title="Running processes"
      >
        <span className="tray-dot tray-dot--live" />
        {count} running
        <span className={`tray-caret${open ? " tray-caret--open" : ""}`}>▾</span>
      </button>

      {open && coords && (
        <div
          ref={popRef}
          className="tray-popover"
          role="dialog"
          aria-label="Running processes"
          style={{ top: coords.top, right: coords.right }}
        >
          <div className="tray-popover-head">
            <span className="tray-popover-title">Running processes</span>
            <button className="tray-stopall" onClick={stopAll}>Stop all</button>
          </div>
          <ul className="tray-list">
            {running.map((id) => {
              const m = metaFor(id);
              return (
                <li key={id} className="tray-item">
                  <span className="tray-item-icon">{m.icon}</span>
                  <button
                    className="tray-item-label"
                    onClick={() => jump(m.panel)}
                    disabled={!m.panel}
                    title={m.panel ? `Go to ${m.label}` : m.label}
                  >
                    <span className="tray-item-name">{m.label}</span>
                    <span className="tray-item-status">
                      <span className="tray-dot tray-dot--live tray-dot--sm" /> active
                    </span>
                  </button>
                  <button className="tray-item-stop" onClick={() => stop(id)} title="Stop">■</button>
                </li>
              );
            })}
          </ul>
        </div>
      )}
    </>
  );
}
