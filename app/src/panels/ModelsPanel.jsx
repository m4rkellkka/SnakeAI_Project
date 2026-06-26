import { useState, useEffect } from "react";
import { invoke } from "@tauri-apps/api/core";
import EmptyState from "../components/EmptyState";
import { formatInteger, formatNumber } from "../utils";

export default function ModelsPanel({ projectRoot, isActive }) {
  const [models, setModels] = useState([]);
  const [isCreating, setIsCreating] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  
  // Form State
  const [algo, setAlgo] = useState("bc");
  const [runName, setRunName] = useState("my_model");
  const [lr, setLr] = useState("0.0005");
  // Behavioral-cloning fields
  const [daggerProb, setDaggerProb] = useState("0.7");
  const [curriculumProb, setCurriculumProb] = useState("0.2");
  // Reinforcement-learning (DQN) fields
  const [gamma, setGamma] = useState("0.95");
  const [warmstartEpisodes, setWarmstartEpisodes] = useState("50");
  
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
        } catch(e) {
          console.error(`Failed to load history for ${cp}:`, e);
        }
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
    
    // Behavioral cloning (train_ai.py) and reinforcement learning (train_rl.py) are
    // separate training methods with different hyperparameters and scripts.
    const args = algo === "dqn"
      ? [
          "-u", "src/train_rl.py",
          "--init-only",
          "--algo", "dqn",
          "--run-name", runName,
          "--lr", lr,
          "--gamma", gamma,
          "--warmstart-episodes", warmstartEpisodes,
        ]
      : [
          "-u", "src/train_ai.py", 
          "--init-only", 
          "--run-name", runName,
          "--lr", lr,
          "--dagger-prob-max", daggerProb,
          "--curriculum-prob", curriculumProb
        ];
    
    setIsLoading(true);
    await invoke("start_process", { id: "init_model", args, cwd: projectRoot });
    
    setTimeout(loadModels, 1500);
    setTimeout(loadModels, 3000);
    setIsCreating(false);
    setRunName("");
  };

  const [confirmDelete, setConfirmDelete] = useState(null);

  const handleDelete = async (path, name) => {
    if (confirmDelete !== path) {
      setConfirmDelete(path);
      setTimeout(() => setConfirmDelete(null), 3000);
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
      <div className="panel-header models-header">
        <h2>Models</h2>
        <button className="btn btn--primary" onClick={() => setIsCreating(true)} disabled={isCreating}>
          + Create New Model
        </button>
      </div>
      <p className="panel-desc">
        Manage your neural network architectures and hyperparameter configurations. Each model has its own isolated folder under <code>model/</code>.
      </p>

      {/* ── Create Form ── */}
      {isCreating && (
        <div className="card create-form-card">
          <div className="card-title">✨ Create New Model</div>
          <form onSubmit={handleCreate}>
            <div className="create-form-grid">
              <div className="form-group">
                <label className="form-label">Training Method</label>
                <select className="input" value={algo} onChange={e => setAlgo(e.target.value)}>
                  <option value="bc">Behavioral Cloning (imitate teacher)</option>
                  <option value="dqn">Reinforcement Learning (Double DQN)</option>
                </select>
              </div>
              <div className="form-group">
                <label className="form-label">Model Folder Name</label>
                <input className="input" value={runName} onChange={e => setRunName(e.target.value)} required placeholder="e.g. big_network_test" />
              </div>
              <div className="form-group">
                <label className="form-label">Learning Rate</label>
                <input className="input" type="number" step="0.0001" value={lr} onChange={e => setLr(e.target.value)} required />
              </div>
              {algo === "dqn" ? (
                <>
                  <div className="form-group">
                    <label className="form-label">Gamma (discount)</label>
                    <input className="input" type="number" step="0.01" value={gamma} onChange={e => setGamma(e.target.value)} required />
                  </div>
                  <div className="form-group">
                    <label className="form-label">Warm-start Episodes</label>
                    <input className="input" type="number" step="1" value={warmstartEpisodes} onChange={e => setWarmstartEpisodes(e.target.value)} required />
                  </div>
                </>
              ) : (
                <>
                  <div className="form-group">
                    <label className="form-label">DAgger Prob Max</label>
                    <input className="input" type="number" step="0.1" value={daggerProb} onChange={e => setDaggerProb(e.target.value)} required />
                  </div>
                  <div className="form-group">
                    <label className="form-label">Curriculum Prob</label>
                    <input className="input" type="number" step="0.1" value={curriculumProb} onChange={e => setCurriculumProb(e.target.value)} required />
                  </div>
                </>
              )}
            </div>
            <div className="create-form-actions">
              <button type="button" className="btn btn--ghost" onClick={() => setIsCreating(false)}>Cancel</button>
              <button type="submit" className="btn btn--primary">Create & Initialize</button>
            </div>
          </form>
        </div>
      )}

      {/* ── Models Grid ── */}
      {isLoading ? (
        <div className="loading-state">Loading models…</div>
      ) : models.length === 0 ? (
        <EmptyState
          icon="🧬"
          title="No models yet"
          desc="Create your first model above to begin training experiments with custom hyperparameters."
        />
      ) : (
        <div className="models-grid">
          {models.map(m => (
            <div key={m.path} className="model-card">
              <div className="model-card-accent" />
              <div className="model-card-body">
                <div className="model-card-header">
                  <span className="model-card-name">
                    {m.name}
                    <span className={`algo-badge algo-badge--${m.algo === "dqn" ? "rl" : "bc"}`}>
                      {m.algo === "dqn" ? "RL · DQN" : "BC"}
                    </span>
                  </span>
                  <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
                    <span className="model-card-date">
                      {m.creation_date
                        ? new Date(m.creation_date * 1000).toLocaleDateString("ru-RU", { day: "2-digit", month: "2-digit", year: "numeric" })
                        : "Unknown"}
                    </span>
                    <button
                      onClick={() => handleDelete(m.path, m.name)}
                      className={`btn-delete${confirmDelete === m.path ? ' btn-delete--confirm' : ''}`}
                      title="Delete model"
                    >
                      {confirmDelete === m.path ? "Sure?" : "🗑"}
                    </button>
                  </div>
                </div>
                
                <div className="model-card-stats">
                  <div>
                    <div className="model-card-stat-label">Games Trained</div>
                    <div className="model-card-stat-value model-card-stat-value--white">{formatInteger(m.n_games || 0)}</div>
                  </div>
                  <div>
                    <div className="model-card-stat-label">Best Eval Avg</div>
                    <div className="model-card-stat-value model-card-stat-value--green">
                      {m.eval_avg_history?.length > 0 ? formatNumber(Math.max(...m.eval_avg_history), 1) : "--"}
                    </div>
                  </div>
                </div>
                
                <div className="model-card-path">
                  <code>{m.path}</code>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
