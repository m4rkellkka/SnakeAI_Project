import { useState, useEffect, useRef } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";

export default function SetupPanel({ envStatus, onRecheck, projectRoot }) {
  const [installing, setInstalling] = useState(false);
  const [done, setDone] = useState(false);
  const [log, setLog] = useState([]);
  const logRef = useRef(null);

  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [log]);

  useEffect(() => {
    const u1 = listen("proc-out:setup-install", (e) => {
      setLog((prev) => [...prev, e.payload]);
    });
    const u2 = listen("proc-done:setup-install", () => {
      setInstalling(false);
      setDone(true);
      setTimeout(() => onRecheck(), 800);
    });
    return () => { u1.then((f) => f()); u2.then((f) => f()); };
  }, [onRecheck]);

  const installDeps = async () => {
    setLog([]);
    setDone(false);
    setInstalling(true);
    try {
      await invoke("start_process", {
        id: "setup-install",
        args: ["-m", "pip", "install", "-r", "requirements.txt"],
        cwd: projectRoot,
      });
    } catch (e) {
      setLog((prev) => [...prev, `Error: ${e}`]);
      setInstalling(false);
    }
  };

  const openUrl = (url) => invoke("open_url", { url });

  const pythonMissing = !envStatus.python_ok;
  const depsMissing = envStatus.python_ok && !envStatus.deps_ok;

  return (
    <div className="setup-screen">
      <div className="setup-card">
        <div className="setup-logo">
          <span className="setup-logo-icon">🐍</span>
          <div>
            <h1 className="setup-title">Snake AI</h1>
            <p className="setup-subtitle">First Run Setup</p>
          </div>
        </div>

        <div className="setup-checks">
          <div className={`setup-check ${envStatus.python_ok ? "setup-check--ok" : "setup-check--fail"}`}>
            <span className="setup-check-icon">{envStatus.python_ok ? "✅" : "❌"}</span>
            <div>
              <div className="setup-check-label">Python 3.8+</div>
              <div className="setup-check-detail">
                {envStatus.python_ok ? envStatus.python_version : "Not found on PATH"}
              </div>
            </div>
          </div>

          <div className={`setup-check ${envStatus.deps_ok ? "setup-check--ok" : "setup-check--fail"}`}>
            <span className="setup-check-icon">{envStatus.deps_ok ? "✅" : "❌"}</span>
            <div>
              <div className="setup-check-label">Python dependencies</div>
              <div className="setup-check-detail">
                {envStatus.deps_ok
                  ? "torch · pygame · numpy · matplotlib"
                  : envStatus.missing_packages.length > 0
                    ? `Missing: ${envStatus.missing_packages.join(", ")}`
                    : "Not checked (Python missing)"}
              </div>
            </div>
          </div>
        </div>

        {pythonMissing && (
          <div className="setup-section">
            <p className="setup-section-title">Install Python 3.8+</p>
            <div className="setup-install-row">
              <code className="setup-cmd">brew install python3</code>
              <button
                className="btn btn--ghost btn--sm"
                onClick={() => openUrl("https://brew.sh")}
              >
                Get Homebrew ↗
              </button>
            </div>
            <div className="setup-divider">or</div>
            <button
              className="btn btn--ghost btn--sm setup-link-btn"
              onClick={() => openUrl("https://www.python.org/downloads/")}
            >
              Download from python.org ↗
            </button>
            <p className="setup-hint">
              After installing Python, click <strong>Check again</strong> below.
            </p>
          </div>
        )}

        {depsMissing && (
          <div className="setup-section">
            <p className="setup-section-title">Install dependencies</p>
            <p className="setup-section-hint">
              Runs <code>pip install -r requirements.txt</code> inside the project folder.
            </p>
          </div>
        )}

        <div className="setup-actions">
          {depsMissing && (
            <button
              className="btn btn--primary"
              onClick={installDeps}
              disabled={installing || !projectRoot}
            >
              {installing ? "⏳ Installing…" : "⬇ Install dependencies"}
            </button>
          )}
          <button
            className="btn btn--ghost"
            onClick={onRecheck}
            disabled={installing}
          >
            ↺ Check again
          </button>
        </div>

        {log.length > 0 && (
          <div className="setup-log" ref={logRef}>
            <div className="setup-log-head">
              {done ? "✅ Done — rechecking…" : "📦 pip output"}
            </div>
            {log.map((line, i) => (
              <div key={i} className="setup-log-line">{line}</div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
