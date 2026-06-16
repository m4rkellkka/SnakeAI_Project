import { useState, useEffect } from "react";
import { invoke } from "@tauri-apps/api/core";
import { formatInteger, formatNumber } from "../utils";

export default function ModelsPanel({ projectRoot, isActive }) {
  const [models, setModels] = useState([]);
  const [isCreating, setIsCreating] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  
  // Form State
  const [runName, setRunName] = useState("my_model");
  const [lr, setLr] = useState("0.0005");
  const [daggerProb, setDaggerProb] = useState("0.7");
  const [curriculumProb, setCurriculumProb] = useState("0.2");
  
  const loadModels = async () => {
    if (!projectRoot) return;
    setIsLoading(true);
    try {
      const list = await invoke("list_checkpoints", { cwd: projectRoot });
      const cpList = list.filter(cp => cp.endsWith("checkpoint_last.pth"));
      
      const loadedModels = [];
      for (const cp of cpList) {
        try {
          const histStr = await invoke("fetch_history", { cwd: projectRoot, checkpointPath: `model/${cp}` });
          const hist = JSON.parse(histStr);
          if (!hist.error) {
            loadedModels.push({ 
              path: cp, 
              name: cp.includes("/") ? cp.split("/")[0] : "Default Run",
              ...hist 
            });
          }
        } catch(e) {}
      }
      // Sort by creation date descending
      loadedModels.sort((a, b) => (b.creation_date || 0) - (a.creation_date || 0));
      setModels(loadedModels);
    } catch(e) {
      console.error(e);
    }
    setIsLoading(false);
  };

  useEffect(() => {
    if (isActive) {
      loadModels();
    }
  }, [projectRoot, isActive]);
  
  const handleCreate = async (e) => {
    e.preventDefault();
    if (!runName) return;
    
    // Call init-only
    const args = [
      "-u", "src/train_ai.py", 
      "--init-only", 
      "--run-name", runName,
      "--lr", lr,
      "--dagger-prob-max", daggerProb,
      "--curriculum-prob", curriculumProb
    ];
    
    setIsLoading(true);
    await invoke("start_process", { id: "init_model", args, cwd: projectRoot });
    
    // Wait a brief moment for files to be written
    setTimeout(loadModels, 1500);
    setIsCreating(false);
    setRunName("");
  };

  const [confirmDelete, setConfirmDelete] = useState(null);

  const handleDelete = async (path, name) => {
    if (confirmDelete !== path) {
      setConfirmDelete(path);
      setTimeout(() => setConfirmDelete(null), 3000); // Reset after 3 seconds
      return;
    }
    
    setIsLoading(true);
    setConfirmDelete(null);
    try {
      await invoke("delete_model", { cwd: projectRoot, path });
      await loadModels();
    } catch (e) {
      console.error("Failed to delete model", e);
      setIsLoading(false);
    }
  };

  return (
    <div className="panel">
      <div className="panel-header" style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div>
          <h2>Models Manager</h2>
        </div>
        <button className="btn btn--primary" onClick={() => setIsCreating(true)} disabled={isCreating}>
          + Create New Model
        </button>
      </div>
      <p className="panel-desc">
        Manage your neural network architectures and hyperparameter configurations.
      </p>

      {isCreating && (
        <div className="card" style={{ marginBottom: "20px", border: "1px solid #3b82f6" }}>
          <div className="card-title">CREATE NEW MODEL</div>
          <form onSubmit={handleCreate} style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "16px" }}>
              <div className="form-group">
                <label className="form-label">Model Folder Name</label>
                <input className="input" value={runName} onChange={e => setRunName(e.target.value)} required placeholder="e.g. big_network_test" />
              </div>
              <div className="form-group">
                <label className="form-label">Learning Rate</label>
                <input className="input" type="number" step="0.0001" value={lr} onChange={e => setLr(e.target.value)} required />
              </div>
              <div className="form-group">
                <label className="form-label">DAgger Prob Max</label>
                <input className="input" type="number" step="0.1" value={daggerProb} onChange={e => setDaggerProb(e.target.value)} required />
              </div>
              <div className="form-group">
                <label className="form-label">Curriculum Prob</label>
                <input className="input" type="number" step="0.1" value={curriculumProb} onChange={e => setCurriculumProb(e.target.value)} required />
              </div>
            </div>
            <div style={{ display: "flex", gap: "12px", justifyContent: "flex-end" }}>
              <button type="button" className="btn btn--ghost" onClick={() => setIsCreating(false)}>Cancel</button>
              <button type="submit" className="btn btn--primary">Create & Initialize</button>
            </div>
          </form>
        </div>
      )}

      {isLoading ? (
        <div style={{ padding: "40px", textAlign: "center", color: "#8b8b9a" }}>Loading models...</div>
      ) : models.length === 0 ? (
        <div style={{ padding: "40px", textAlign: "center", color: "#8b8b9a" }}>No models found. Create one above!</div>
      ) : (
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "20px" }}>
          {models.map(m => (
            <div key={m.path} className="card" style={{ display: "flex", flexDirection: "column", gap: "12px", position: "relative" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <div className="card-title" style={{ fontSize: "16px", color: "#fff", margin: 0 }}>{m.name}</div>
                <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
                  <div style={{ fontSize: "12px", color: "#8b8b9a" }}>
                    {m.creation_date ? new Date(m.creation_date * 1000).toLocaleDateString("ru-RU", { day: "2-digit", month: "2-digit", year: "numeric" }) : "Unknown Date"}
                  </div>
                  <button 
                    onClick={() => handleDelete(m.path, m.name)}
                    style={{ background: "transparent", border: "none", cursor: "pointer", fontSize: "16px", padding: "4px", borderRadius: "4px", color: confirmDelete === m.path ? "#ef4444" : "inherit" }}
                    title="Delete model"
                    className="btn-link"
                  >
                    {confirmDelete === m.path ? "Sure?" : "❌"}
                  </button>
                </div>
              </div>
              
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "12px", background: "rgba(0,0,0,0.2)", padding: "12px", borderRadius: "8px" }}>
                <div>
                  <div style={{ fontSize: "11px", color: "#8b8b9a", textTransform: "uppercase" }}>Games Trained</div>
                  <div style={{ fontSize: "20px", fontWeight: "bold", color: "#e4e4e7" }}>{formatInteger(m.n_games || 0)}</div>
                </div>
                <div>
                  <div style={{ fontSize: "11px", color: "#8b8b9a", textTransform: "uppercase" }}>Best Eval Avg</div>
                  <div style={{ fontSize: "20px", fontWeight: "bold", color: "#10b981" }}>
                    {m.eval_avg_history?.length > 0 ? formatNumber(Math.max(...m.eval_avg_history), 1) : "--"}
                  </div>
                </div>
              </div>
              
              <div style={{ fontSize: "13px", color: "#8b8b9a" }}>
                <code style={{ background: "rgba(0,0,0,0.3)", padding: "2px 6px", borderRadius: "4px" }}>{m.path}</code>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
