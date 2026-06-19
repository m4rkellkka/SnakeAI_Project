import { useState, useEffect } from 'react';
import { invoke } from '@tauri-apps/api/core';

export default function SettingsPanel() {
  const [envInfo, setEnvInfo] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    (async () => {
      try {
        const result = await invoke('check_environment');
        setEnvInfo(result);
      } catch (e) {
        console.error('Failed to check environment:', e);
      }
      setLoading(false);
    })();
  }, []);

  if (loading) {
    return (
      <div className="panel">
        <div className="panel-header">
          <h2>Settings</h2>
        </div>
        <div className="loading-state">Loading environment info…</div>
      </div>
    );
  }

  return (
    <div className="panel">
      <div className="panel-header">
        <h2>Settings</h2>
      </div>
      <p className="panel-desc">
        Environment configuration and system information for your Snake AI workspace.
      </p>

      <div className="settings-grid">
        {/* Environment Section */}
        <div className="settings-section">
          <div className="settings-section-title">🐍 Python Environment</div>

          <div className="settings-row">
            <div className="settings-label">Python Path</div>
            <div className="settings-value">
              {envInfo?.python_path || 'Not detected'}
            </div>
          </div>

          <div className="settings-row">
            <div className="settings-label">Python Version</div>
            <div className="settings-value">
              {envInfo?.python_version || 'Unknown'}
            </div>
          </div>

          <div className="settings-row">
            <div className="settings-label">Dependencies Status</div>
            <div className="settings-value">
              {envInfo?.missing_packages?.length === 0
                ? '✓ All packages installed'
                : `⚠ Missing: ${envInfo?.missing_packages?.join(', ') || 'unknown'}`
              }
            </div>
          </div>
        </div>

        {/* Project Section */}
        <div className="settings-section">
          <div className="settings-section-title">📁 Project</div>

          <div className="settings-row">
            <div className="settings-label">Project Directory</div>
            <div className="settings-value">
              {envInfo?.project_root || 'Unknown'}
            </div>
          </div>

          <div className="settings-row">
            <div className="settings-label">Version</div>
            <div className="settings-value">1.0.0</div>
          </div>
        </div>

        {/* About Section */}
        <div className="settings-section">
          <div className="settings-section-title">ℹ️ About</div>

          <div className="settings-row">
            <div className="settings-label">Application</div>
            <div className="settings-value">Snake AI Desktop</div>
            <div className="settings-hint">
              Reinforcement learning environment for training neural networks to play Snake.
              Built with Tauri + React + Python.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
