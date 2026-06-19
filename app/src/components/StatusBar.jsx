export default function StatusBar({ activePanel, running, projectRoot }) {
  const panelNames = {
    train: 'Training',
    watch: 'Watch AI',
    bench: 'Benchmark',
    models: 'Models',
    settings: 'Settings',
  };

  // Count running processes
  const runningProcs = Object.values(running || {}).filter(Boolean);
  const anyRunning = runningProcs.length > 0;

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
        {anyRunning && (
          <>
            <div className="status-bar-item">
              <span className="status-bar-item-icon">⚡</span>
              <span className="status-bar-item-value status-bar-item-value--green">
                {runningProcs.length} process{runningProcs.length > 1 ? 'es' : ''} running
              </span>
            </div>
            <div className="status-bar-divider" />
          </>
        )}
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
