export default function EmptyState({ icon, title, desc, children }) {
  return (
    <div className="empty-state">
      <div className="empty-state-icon">{icon || '📦'}</div>
      <div className="empty-state-title">{title || 'Nothing here yet'}</div>
      <div className="empty-state-desc">{desc || ''}</div>
      {children}
    </div>
  );
}
