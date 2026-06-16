import { useState, useEffect, useCallback } from "react";
import { invoke } from "@tauri-apps/api/core";
import TrainPanel from "./panels/TrainPanel";
import WatchPanel from "./panels/WatchPanel";
import BenchmarkPanel from "./panels/BenchmarkPanel";
import ModelsPanel from "./panels/ModelsPanel";
import SetupPanel from "./panels/SetupPanel";
import "./App.css";

const NAV = [
  { id: "train",     label: "Train",     icon: "⚡" },
  { id: "watch",     label: "Watch AI",  icon: "👁" },
  { id: "benchmark", label: "Benchmark", icon: "📊" },
  { id: "models",    label: "Models",    icon: "📁" },
];

export default function App() {
  const [panel, setPanel] = useState("train");
  const [projectRoot, setProjectRoot] = useState("");
  const [running, setRunning] = useState([]);
  const [envStatus, setEnvStatus] = useState(null);

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

  const isPanelRunning = (panelId) => {
    if (panelId === "train" || panelId === "benchmark") {
      return running.includes(panelId);
    }
    if (panelId === "watch") {
      return running.some(p => p.startsWith("watch-") || p === "play-manual");
    }
    return false;
  };

  if (envStatus === null) {
    return (
      <div className="setup-screen">
        <div className="setup-checking">
          <span className="setup-logo-icon">🐍</span>
          <p className="setup-checking-text">Checking environment…</p>
        </div>
      </div>
    );
  }

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

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="sidebar-logo">
          <span className="sidebar-logo-icon">🐍</span>
          <span className="sidebar-logo-text">Snake AI</span>
        </div>
        <nav className="sidebar-nav">
          {NAV.map((item) => (
            <button
              key={item.id}
              className={`nav-item ${panel === item.id ? "nav-item--active" : ""}`}
              onClick={() => setPanel(item.id)}
            >
              <span className="nav-icon">{item.icon}</span>
              <span>{item.label}</span>
              {isPanelRunning(item.id) && <span className="nav-dot" />}
            </button>
          ))}
        </nav>
        <div className="sidebar-footer">
          <span className="version-tag">v1.0</span>
          <button
            className="btn-quit"
            onClick={async () => {
              if (running.length > 0 && !window.confirm("Processes are still running. Quit anyway?")) return;
              await invoke("quit_app");
            }}
          >
            ⏻ Quit
          </button>
        </div>
      </aside>

      <main className="main">
        <div style={{ display: panel === "train" ? "block" : "none", height: "100%" }}>
          <TrainPanel
            projectRoot={projectRoot}
            isRunning={running.includes("train")}
            isActive={panel === "train"}
          />
        </div>
        
        <div style={{ display: panel === "watch" ? "block" : "none", height: "100%" }}>
          <WatchPanel
            projectRoot={projectRoot}
            runningList={running}
            isActive={panel === "watch"}
          />
        </div>
        
        <div style={{ display: panel === "benchmark" ? "block" : "none", height: "100%" }}>
          <BenchmarkPanel
            projectRoot={projectRoot}
            isRunning={running.includes("benchmark")}
            isActive={panel === "benchmark"}
          />
        </div>
        
        <div style={{ display: panel === "models" ? "block" : "none", height: "100%" }}>
          <ModelsPanel
            projectRoot={projectRoot}
            isActive={panel === "models"}
          />
        </div>
      </main>
    </div>
  );
}
