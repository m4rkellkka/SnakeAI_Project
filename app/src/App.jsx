import { useState, useEffect, useCallback } from "react";
import { invoke } from "@tauri-apps/api/core";
import { ToastProvider, useToast } from "./components/Toast";
import { usePersistedState } from "./hooks/usePersistedState";
import StatusBar from "./components/StatusBar";
import TrainPanel from "./panels/TrainPanel";
import PlayPanel from "./panels/PlayPanel";
import BenchmarkPanel from "./panels/BenchmarkPanel";
import ModelsPanel from "./panels/ModelsPanel";
import SetupPanel from "./panels/SetupPanel";
import SettingsPanel from "./panels/SettingsPanel";
import "./App.css";

const NAV_SECTIONS = [
  {
    label: "Training",
    items: [
      { id: "train", label: "Train", icon: "⚡" },
      { id: "play", label: "Play", icon: "🎮" },
    ],
  },
  {
    label: "Analytics",
    items: [
      { id: "benchmark", label: "Benchmark", icon: "📊" },
      { id: "models", label: "Models", icon: "📁" },
    ],
  },
  {
    label: "System",
    items: [
      { id: "settings", label: "Settings", icon: "⚙" },
    ],
  },
];

// Shows a one-time toast after an (external) full-page reload so the user knows
// their state was restored rather than silently swapped. Lives inside
// ToastProvider so it can use the toast context.
function ReloadNotice() {
  const toast = useToast();
  useEffect(() => {
    // sessionStorage survives a page reload but not an app relaunch — exactly
    // how we tell "the WebView reloaded under us" apart from a fresh launch.
    if (sessionStorage.getItem("snakeai:mounted")) {
      toast("↻ Интерфейс перезагружен — состояние восстановлено", "info");
    }
    sessionStorage.setItem("snakeai:mounted", "1");
  }, [toast]);
  return null;
}

export default function App() {
  const [panel, setPanel] = usePersistedState("panel", "train");
  const [projectRoot, setProjectRoot] = useState("");
  const [running, setRunning] = useState([]);
  const [envStatus, setEnvStatus] = useState(null);
  const [collapsed, setCollapsed] = useState(false);

  // The Watch AI tab was folded into Play — migrate a persisted "watch" selection
  // so returning users don't land on a now-removed (blank) panel.
  useEffect(() => {
    if (panel === "watch") setPanel("play");
  }, [panel, setPanel]);

  const checkEnv = useCallback(async () => {
    try {
      const status = await invoke("check_environment");
      setEnvStatus(status);
    } catch (e) {
      setEnvStatus({ python_ok: false, python_version: "", deps_ok: false, missing_packages: [] });
    }
  }, []);

  useEffect(() => {
    invoke("get_project_root").then(setProjectRoot);
    checkEnv();

    const interval = setInterval(async () => {
      try {
        const procs = await invoke("list_processes");
        setRunning(procs);
      } catch (e) {
        console.error("Failed to list processes", e);
      }
    }, 1000);

    return () => clearInterval(interval);
  }, [checkEnv]);

  // Diagnostic: in dev, log when (and ideally why) Vite forces a full reload,
  // so we can confirm the trigger from the console and tighten watch.ignored.
  useEffect(() => {
    const hot = import.meta.hot;
    if (!hot) return;
    const onReload = (payload) => {
      console.warn("[SnakeAI] Vite triggered a full page reload", payload);
    };
    hot.on("vite:beforeFullReload", onReload);
    return () => hot.off?.("vite:beforeFullReload", onReload);
  }, []);

  const isPanelRunning = (panelId) => {
    if (panelId === "train" || panelId === "benchmark") {
      return running.includes(panelId);
    }
    if (panelId === "play") {
      return running.includes("play-watch");
    }
    return false;
  };

  // Build a running map for the status bar
  const runningMap = {};
  running.forEach(p => { runningMap[p] = true; });

  const [projectRootInput, setProjectRootInput] = useState("");
  const [projectRootError, setProjectRootError] = useState("");

  const handleSetProjectRoot = async () => {
    setProjectRootError("");
    try {
      const resolved = await invoke("set_project_root", { path: projectRootInput });
      setProjectRoot(resolved);
      checkEnv();
    } catch (e) {
      setProjectRootError(String(e));
    }
  };

  /* ── Loading screen ── */
  if (envStatus === null || (projectRoot === "" && envStatus === null)) {
    return (
      <div className="setup-screen">
        <div className="setup-checking">
          <span className="setup-logo-icon">🐍</span>
          <p className="setup-checking-text">Checking environment…</p>
        </div>
      </div>
    );
  }

  /* ── No project root ── */
  if (projectRoot === "") {
    return (
      <div className="setup-screen">
        <div className="setup-card">
          <div className="setup-logo">
            <span className="setup-logo-icon">🐍</span>
            <div>
              <h1 className="setup-title">Snake AI</h1>
              <p className="setup-subtitle">Project folder not found</p>
            </div>
          </div>
          <div className="setup-section">
            <p className="setup-section-title">Select project folder</p>
            <p className="setup-section-hint">
              Paste the path to your <code>SnakeAI_Project</code> folder
              (the one containing <code>src/train_ai.py</code>).
            </p>
            <input
              className="input"
              style={{ width: "100%", marginTop: "8px" }}
              placeholder="/Users/you/Documents/GitHub/SnakeAI_Project"
              value={projectRootInput}
              onChange={e => setProjectRootInput(e.target.value)}
              onKeyDown={e => e.key === "Enter" && handleSetProjectRoot()}
            />
            {projectRootError && (
              <p style={{ color: "var(--red)", fontSize: "13px", marginTop: "8px" }}>{projectRootError}</p>
            )}
          </div>
          <div className="setup-actions">
            <button className="btn btn--primary" onClick={handleSetProjectRoot} disabled={!projectRootInput}>
              Set project folder
            </button>
          </div>
        </div>
      </div>
    );
  }

  /* ── Setup screen ── */
  if (!envStatus.python_ok || !envStatus.deps_ok) {
    return (
      <SetupPanel
        envStatus={envStatus}
        onRecheck={checkEnv}
        onInstallComplete={() =>
          setEnvStatus((prev) => ({ ...prev, deps_ok: true, python_ok: true, missing_packages: [] }))
        }
        projectRoot={projectRoot}
      />
    );
  }

  /* ── Main app ── */
  return (
    <ToastProvider>
      <ReloadNotice />
      <div className="app">
        <aside className={`sidebar${collapsed ? ' sidebar--collapsed' : ''}`}>
          <div className="sidebar-logo">
            <span className="sidebar-logo-icon">🐍</span>
            <span className="sidebar-logo-text">Snake AI</span>
            <button
              className="sidebar-toggle"
              onClick={() => setCollapsed(!collapsed)}
              title={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
            >
              {collapsed ? '›' : '‹'}
            </button>
          </div>

          <nav className="sidebar-nav">
            {NAV_SECTIONS.map((section, si) => (
              <div key={section.label}>
                <div className="nav-section-label">{section.label}</div>
                {section.items.map((item) => (
                  <button
                    key={item.id}
                    className={`nav-item${panel === item.id ? ' nav-item--active' : ''}`}
                    onClick={() => setPanel(item.id)}
                    title={collapsed ? item.label : undefined}
                  >
                    <span className="nav-icon">{item.icon}</span>
                    <span className="nav-item-label">{item.label}</span>
                    {isPanelRunning(item.id) && <span className="nav-dot" />}
                  </button>
                ))}
                {si < NAV_SECTIONS.length - 1 && <div className="sidebar-divider" />}
              </div>
            ))}
          </nav>

          <div className="sidebar-footer">
            <span className="version-tag">v2.0.0</span>
            <button
              className="btn-quit"
              onClick={async () => {
                if (running.length > 0 && !window.confirm("Processes are still running. Quit anyway?")) return;
                await invoke("quit_app");
              }}
            >
              <span className="btn-quit-icon">⏻</span> <span className="btn-quit-text">Quit</span>
            </button>
          </div>
        </aside>

        <main className="main">
          <StatusBar
            activePanel={panel}
            running={runningMap}
            runningList={running}
            projectRoot={projectRoot}
            onNavigate={setPanel}
          />

          <div className="main-content">
            <div style={{ display: panel === "train" ? "block" : "none" }}>
              <TrainPanel
                projectRoot={projectRoot}
                isRunning={running.includes("train")}
                isActive={panel === "train"}
              />
            </div>

            <div style={{ display: panel === "play" ? "block" : "none" }}>
              <PlayPanel
                isActive={panel === "play"}
                projectRoot={projectRoot}
                runningList={running}
              />
            </div>

            <div style={{ display: panel === "benchmark" ? "block" : "none" }}>
              <BenchmarkPanel
                projectRoot={projectRoot}
                isRunning={running.includes("benchmark")}
                isActive={panel === "benchmark"}
              />
            </div>

            <div style={{ display: panel === "models" ? "block" : "none" }}>
              <ModelsPanel
                projectRoot={projectRoot}
                isActive={panel === "models"}
              />
            </div>

            {panel === "settings" && <SettingsPanel />}
          </div>
        </main>
      </div>
    </ToastProvider>
  );
}
