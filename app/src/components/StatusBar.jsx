import ProcessTray from "./ProcessTray";

export default function StatusBar({ activePanel, running, runningList, projectRoot, onNavigate }) {
  const panelNames = {
    train: 'Training',
    watch: 'Watch AI',
    benchmark: 'Benchmark',
    models: 'Models',
    settings: 'Settings',
  };

  return (
    <div className="status-bar">
      <div className="status-bar-left">
        <div className="status-bar-breadcrumb">
          <span>Snake AI</span>
          <span style={{ opacity: 0.3 }}>›</span>
          <span className="status-bar-breadcrumb-active">
            {panelNames[activePanel] || 'Dashboard'}
          </span>
        </div>
      </div>
      <div className="status-bar-right">
        <ProcessTray running={runningList || []} onNavigate={onNavigate} />
        <div className="status-bar-divider" />
        <div className="status-bar-item">
          <span className="status-bar-item-icon">📁</span>
          <span className="status-bar-item-value">
            {projectRoot ? projectRoot.split('/').pop() || projectRoot : '—'}
          </span>
        </div>
      </div>
    </div>
  );
}
