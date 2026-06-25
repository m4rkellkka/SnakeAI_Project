import { useState, useEffect } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { useSettings } from '../components/SettingsProvider';

const DENSITIES = [
  { id: 'comfortable', label: 'Comfortable', hint: 'Roomy spacing' },
  { id: 'compact', label: 'Compact', hint: 'More on screen' },
];

export default function SettingsPanel() {
  const { accent, setAccent, accents, density, setDensity, reducedMotion, setReducedMotion } = useSettings();
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

  return (
    <div className="panel">
      <div className="panel-header">
        <h2>Settings</h2>
      </div>
      <p className="panel-desc">
        Personalize the workspace and inspect your environment. Appearance changes apply
        instantly and are remembered between sessions.
      </p>

      <div className="settings-grid">
        {/* ── Appearance (interactive) ── */}
        <div className="settings-section">
          <div className="settings-section-title">🎨 Appearance</div>

          <div className="settings-control">
            <div className="settings-control-text">
              <div className="settings-control-label">Accent color</div>
              <div className="settings-hint">Recolors primary actions, selection, and live indicators.</div>
            </div>
            <div className="swatch-row" role="radiogroup" aria-label="Accent color">
              {accents.map((a) => (
                <button
                  key={a.id}
                  type="button"
                  role="radio"
                  aria-checked={accent === a.id}
                  aria-label={a.label}
                  title={a.label}
                  className={`swatch${accent === a.id ? ' swatch--active' : ''}`}
                  style={{ '--swatch': a.base }}
                  onClick={() => setAccent(a.id)}
                />
              ))}
            </div>
          </div>

          <div className="settings-control">
            <div className="settings-control-text">
              <div className="settings-control-label">Density</div>
              <div className="settings-hint">Tighten spacing to fit more on smaller screens.</div>
            </div>
            <div className="segmented" role="radiogroup" aria-label="Density">
              {DENSITIES.map((d) => (
                <button
                  key={d.id}
                  type="button"
                  role="radio"
                  aria-checked={density === d.id}
                  className={`segmented-option${density === d.id ? ' segmented-option--active' : ''}`}
                  onClick={() => setDensity(d.id)}
                  title={d.hint}
                >
                  {d.label}
                </button>
              ))}
            </div>
          </div>

          <div className="settings-control">
            <div className="settings-control-text">
              <div className="settings-control-label">Reduce motion</div>
              <div className="settings-hint">Disable count-ups, pulses, and entrance animations.</div>
            </div>
            <button
              type="button"
              role="switch"
              aria-checked={reducedMotion}
              className={`toggle${reducedMotion ? ' toggle--on' : ''}`}
              onClick={() => setReducedMotion(!reducedMotion)}
            >
              <span className="toggle-knob" />
            </button>
          </div>
        </div>

        {/* ── Environment (read-only) ── */}
        <div className="settings-section">
          <div className="settings-section-title">🐍 Python Environment</div>

          {loading ? (
            <div className="settings-skeleton-group">
              <div className="skeleton skeleton--line" />
              <div className="skeleton skeleton--line" />
              <div className="skeleton skeleton--line" />
            </div>
          ) : (
            <>
              <div className="settings-row">
                <div className="settings-label">Python Path</div>
                <div className="settings-value">{envInfo?.python_path || 'Not detected'}</div>
              </div>
              <div className="settings-row">
                <div className="settings-label">Python Version</div>
                <div className="settings-value">{envInfo?.python_version || 'Unknown'}</div>
              </div>
              <div className="settings-row">
                <div className="settings-label">Dependencies Status</div>
                <div className="settings-value">
                  {envInfo?.missing_packages?.length === 0
                    ? '✓ All packages installed'
                    : `⚠ Missing: ${envInfo?.missing_packages?.join(', ') || 'unknown'}`}
                </div>
              </div>
            </>
          )}
        </div>

        {/* ── Project (read-only) ── */}
        <div className="settings-section">
          <div className="settings-section-title">📁 Project</div>
          <div className="settings-row">
            <div className="settings-label">Project Directory</div>
            <div className="settings-value">{envInfo?.project_root || 'Unknown'}</div>
          </div>
          <div className="settings-row">
            <div className="settings-label">Version</div>
            <div className="settings-value">1.0.0</div>
          </div>
        </div>

        {/* ── About ── */}
        <div className="settings-section">
          <div className="settings-section-title">ℹ️ About</div>
          <div className="settings-row">
            <div className="settings-label">Application</div>
            <div className="settings-value">Snake AI Desktop</div>
            <div className="settings-hint">
              Behavioral-cloning &amp; reinforcement-learning environment for training neural
              networks to play Snake. Built with Tauri + React + Python.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
