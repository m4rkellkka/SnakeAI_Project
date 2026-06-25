import { useEffect, useRef, useState } from "react";
import { useSettings } from "./SettingsProvider";

const easeOutQuart = (t) => 1 - Math.pow(1 - t, 4);

/**
 * Counts a metric up to its new value over ~500ms with an ease-out curve, so
 * live numbers feel responsive rather than snapping. Honors the reduced-motion
 * preference (and OS setting, via the same flag) by rendering the final value
 * immediately. `format` receives the interpolated number and returns a string.
 */
export default function AnimatedNumber({ value, format = (n) => String(n), duration = 500, className }) {
  const { reducedMotion } = useSettings();
  const numeric = typeof value === "number" && Number.isFinite(value);
  const [display, setDisplay] = useState(numeric ? value : 0);
  const fromRef = useRef(numeric ? value : 0);
  const rafRef = useRef(null);

  useEffect(() => {
    if (!numeric) return;
    if (reducedMotion) {
      fromRef.current = value;
      setDisplay(value);
      return;
    }
    const from = fromRef.current;
    const to = value;
    if (from === to) return;
    const start = performance.now();

    const tick = (now) => {
      const t = Math.min(1, (now - start) / duration);
      const eased = easeOutQuart(t);
      const current = from + (to - from) * eased;
      setDisplay(current);
      if (t < 1) {
        rafRef.current = requestAnimationFrame(tick);
      } else {
        fromRef.current = to;
      }
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafRef.current);
  }, [value, duration, reducedMotion, numeric]);

  if (!numeric) {
    return <span className={className}>{value ?? "—"}</span>;
  }
  return <span className={className}>{format(display)}</span>;
}
