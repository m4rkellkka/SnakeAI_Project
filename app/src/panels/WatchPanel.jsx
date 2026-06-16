import { useState, useEffect, useRef } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";
import { formatCheckpoint, formatNumber, formatInteger } from "../utils";

const CHART_STYLE = {
  contentStyle: { background: "rgba(17,17,24,0.9)", border: "1px solid rgba(30,30,46,0.8)", borderRadius: 12, fontSize: 12, backdropFilter: "blur(10px)" },
  labelStyle: { color: "#8b8b9a" },
};

const MODES = [
  {
    id: "watch-pretrained",
    title: "Watch Pretrained",
    desc: "Load pretrained.pth and watch the best model play.",
    args: ["-u", "src/train_ai.py", "--watch", "--pretrained"],
    icon: "🏆",
    color: "green",
  },
  {
    id: "watch-checkpoint",
    title: "Watch Checkpoint",
    desc: "Load a specific checkpoint from your training run.",
    args: ["-u", "src/train_ai.py", "--watch"], // We'll append --checkpoint later
    icon: "💾",
    color: "blue",
    needsCheckpoint: true,
  },
  {
    id: "watch-teacher",
    title: "Watch Teacher",
    desc: "Watch the perfect Hamiltonian-cycle teacher.",
    args: ["-u", "src/teacher.py"],
    icon: "📐",
    color: "orange",
  },
  {
    id: "play-manual",
    title: "Play Manually",
    desc: "Control the snake yourself with arrow keys.",
    args: ["-u", "src/play_manual.py"],
    icon: "🎮",
    color: "purple",
  },
];

export default function WatchPanel({ projectRoot, runningList, isActive }) {
  const [checkpoint, setCheckpoint] = useState("checkpoint_best.pth");
  const [checkpointList, setCheckpointList] = useState(["checkpoint_best.pth"]);
  const [log, setLog]         = useState([]);
  const [liveData, setLiveData] = useState([]);
  const [history, setHistory] = useState(null);
  
  const logBodyRef = useRef(null);
  const isUserScrolling = useRef(false);

  useEffect(() => {
    if (projectRoot && isActive) {
      invoke("list_checkpoints", { cwd: projectRoot }).then((list) => {
        if (list.length > 0) {
          const filtered = list.filter(cp => cp.endsWith("checkpoint_best.pth"));
          setCheckpointList(filtered);
          if (!filtered.includes(checkpoint) && filtered.length > 0) {
            setCheckpoint(filtered[0]);
          }
        }
      });
    }
  }, [projectRoot]);

  useEffect(() => {
    if (projectRoot && checkpoint) {
      invoke("fetch_history", { cwd: projectRoot, checkpointPath: `model/${checkpoint}` })
        .then(histStr => {
          const hist = JSON.parse(histStr);
          if (hist.error) {
            setHistory(null);
          } else {
            const evalData = [];
            if (hist.eval_games_history) {
              for (let i = 0; i < hist.eval_games_history.length; i++) {
                evalData.push({
                  game: hist.eval_games_history[i],
                  avg: hist.eval_avg_history[i],
                  max: hist.eval_max_history[i],
                });
              }
            }
            setHistory({
              n_games: hist.n_games || 0,
              total_steps: hist.total_steps || 0,
              creation_date: hist.creation_date || null,
              evalData,
            });
          }
        })
        .catch(() => setHistory(null));
    }
  }, [projectRoot, checkpoint]);

  useEffect(() => {
    const unlistens = MODES.flatMap((m) => [
      listen(`proc-out:${m.id}`, (e) => {
        const line = e.payload;
        setLog((p) => [...p, line].slice(-300));
        
        const match = /Game (\d+)\/\d+ \| Score: (\d+)/.exec(line);
        if (match) {
          setLiveData((p) => [...p, { game: +match[1], score: +match[2] }]);
        }
      }),
      listen(`proc-done:${m.id}`, () => {}),
    ]);

    return () => { unlistens.forEach((p) => p.then((f) => f())); };
  }, []);

  useEffect(() => {
    if (logBodyRef.current && !isUserScrolling.current) {
      logBodyRef.current.scrollTop = logBodyRef.current.scrollHeight;
    }
  }, [log]);

  const handleScroll = () => {
    if (!logBodyRef.current) return;
    const { scrollTop, scrollHeight, clientHeight } = logBodyRef.current;
    isUserScrolling.current = scrollHeight - scrollTop - clientHeight > 10;
  };

  const launch = async (mode) => {
    if (!projectRoot) return;
    const args = [...mode.args];
    if (mode.needsCheckpoint) {
      args.push("--checkpoint", checkpoint);
    }
    setLog([]);
    setLiveData([]);
    await invoke("start_process", { id: mode.id, args, cwd: projectRoot });
    isUserScrolling.current = false;
  };

  const stop = async (mode) => {
    await invoke("stop_process", { id: mode.id });
  };

  const running = {};
  if (runningList) {
    for (const m of MODES) {
      running[m.id] = runningList.includes(m.id);
    }
  }
  const isAnyRunning = Object.values(running).some(Boolean);

  return (
    <div className="panel">
      <div className="panel-header">
        <h2>Watch AI</h2>
      </div>
      <p className="panel-desc">
        Each button opens a separate Pygame window. The Tauri dashboard stays open.
      </p>

      <div className="watch-grid">
        {MODES.map((m) => (
          <div key={m.id} className={`watch-card watch-card--${m.color}`}>
            <span className="watch-icon">{m.icon}</span>
            <h3>{m.title}</h3>
            <p>{m.desc}</p>
            {m.needsCheckpoint && (
              <div className="form-group" style={{ marginTop: 'auto', marginBottom: '8px' }}>
                <select className="input select" value={checkpoint} onChange={(e) => setCheckpoint(e.target.value)} disabled={isAnyRunning} style={{ fontSize: 12, padding: "6px 8px" }}>
                  {checkpointList.map(cp => <option key={cp} value={cp}>{formatCheckpoint(cp)}</option>)}
                </select>
              </div>
            )}
            <div className="watch-actions">
              {running[m.id] ? (
                <button className="btn btn--stop btn--sm" onClick={() => stop(m)}>■ Stop</button>
              ) : (
                <button className="btn btn--ghost btn--sm" onClick={() => launch(m)} disabled={!projectRoot || (m.needsCheckpoint && !checkpoint)}>
                  ▶ Launch
                </button>
              )}
              {running[m.id] && <span className="running-label">● Running</span>}
            </div>
          </div>
        ))}
      </div>

      <div style={{ display: "flex", gap: "20px", marginTop: "20px" }}>
        <div className="card" style={{ flex: 2 }}>
          <div className="card-title" style={{ marginBottom: "16px" }}>LIVE SCORE PER GAME</div>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={liveData} margin={{ top: 10, right: 20, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(30,30,46,0.6)" />
              <XAxis dataKey="game" stroke="#8b8b9a" tick={{ fontSize: 10 }} />
              <YAxis stroke="#8b8b9a" tick={{ fontSize: 10 }} domain={[0, 253]} />
              <Tooltip {...CHART_STYLE} />
              <Line type="monotone" dataKey="score" stroke="#10b981" strokeWidth={2} dot={true} name="Score" />
            </LineChart>
          </ResponsiveContainer>
        </div>
        
        <div style={{ flex: 1, display: "grid", gridTemplateColumns: "1fr 1fr", gap: "16px" }}>
          <div className="card" style={{ display: "flex", flexDirection: "column", justifyContent: "center", alignItems: "center" }}>
            <div style={{ fontSize: "11px", color: "#8b8b9a", textTransform: "uppercase", letterSpacing: "1px", marginBottom: "8px" }}>Games Trained</div>
            <div style={{ fontSize: "28px", fontWeight: "700", color: "#e4e4e7" }}>{history ? formatInteger(history.n_games) : "--"}</div>
          </div>
          <div className="card" style={{ display: "flex", flexDirection: "column", justifyContent: "center", alignItems: "center" }}>
            <div style={{ fontSize: "11px", color: "#8b8b9a", textTransform: "uppercase", letterSpacing: "1px", marginBottom: "8px" }}>Total Steps</div>
            <div style={{ fontSize: "28px", fontWeight: "700", color: "#f59e0b" }}>{history ? formatInteger(history.total_steps) : "--"}</div>
          </div>
          <div className="card" style={{ display: "flex", flexDirection: "column", justifyContent: "center", alignItems: "center" }}>
            <div style={{ fontSize: "11px", color: "#8b8b9a", textTransform: "uppercase", letterSpacing: "1px", marginBottom: "8px" }}>Created At</div>
            <div style={{ fontSize: "20px", fontWeight: "700", color: "#3b82f6", textAlign: "center" }}>
              {history && history.creation_date ? new Date(history.creation_date * 1000).toLocaleDateString("ru-RU", { day: "2-digit", month: "2-digit", year: "numeric" }) : "--"}
            </div>
          </div>
          <div className="card" style={{ display: "flex", flexDirection: "column", justifyContent: "center", alignItems: "center" }}>
            <div style={{ fontSize: "11px", color: "#8b8b9a", textTransform: "uppercase", letterSpacing: "1px", marginBottom: "8px" }}>Live Mean Score</div>
            <div style={{ fontSize: "28px", fontWeight: "700", color: "#10b981" }}>
              {liveData.length > 0 ? formatNumber(liveData.reduce((s, d) => s + d.score, 0) / liveData.length, 1) : "--"}
            </div>
          </div>
        </div>
      </div>

      {log.length > 0 && (
        <div className="log-card" style={{ marginTop: "20px" }}>
          <div className="log-head">
            <span className="log-head-title">Output</span>
            <button className="btn-link" onClick={() => { setLog([]); setLiveData([]); isUserScrolling.current = false; }}>Clear</button>
          </div>
          <div className="log-body" ref={logBodyRef} onScroll={handleScroll}>
            {log.map((l, i) => <div key={i} className="log-line">{l}</div>)}
          </div>
        </div>
      )}
    </div>
  );
}
