import { useState, useEffect, useRef } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import EmptyState from "../components/EmptyState";
import { formatCheckpoint, formatNumber, formatInteger } from "../utils";

const CHART_STYLE = {
  contentStyle: { background: "rgba(12,12,18,0.95)", border: "1px solid rgba(63,63,70,0.4)", borderRadius: 10, fontSize: 12, backdropFilter: "blur(12px)" },
  labelStyle: { color: "#94a3b8" },
};

const MODES = [
  {
    id: "watch-pretrained",
    title: "Watch Pretrained",
    desc: "Load the best pretrained model and watch it play Snake at near-perfect level.",
    args: ["-u", "src/train_ai.py", "--watch", "--pretrained"],
    icon: "🏆",
    color: "green",
  },
  {
    id: "watch-checkpoint",
    title: "Watch Checkpoint",
    desc: "Load a specific checkpoint from your training run to evaluate progress.",
    args: ["-u", "src/train_ai.py", "--watch"],
    icon: "💾",
    color: "blue",
    needsCheckpoint: true,
  },
  {
    id: "watch-teacher",
    title: "Watch Teacher",
    desc: "Watch the perfect Hamiltonian-cycle teacher achieve 100% score every time.",
    args: ["-u", "src/teacher.py"],
    icon: "📐",
    color: "orange",
  },
  {
    id: "play-manual",
    title: "Play Manually",
    desc: "Take control of the snake yourself using arrow keys in a Pygame window.",
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
  const [logCollapsed, setLogCollapsed] = useState(false);
  
  const logBodyRef = useRef(null);
  const isUserScrolling = useRef(false);
  const liveMeanRef = useRef({ sum: 0, count: 0 });
  const [liveMean, setLiveMean] = useState(null);

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
  }, [projectRoot, isActive]);

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
          const score = +match[2];
          liveMeanRef.current.sum += score;
          liveMeanRef.current.count += 1;
          setLiveMean(liveMeanRef.current.sum / liveMeanRef.current.count);
          setLiveData((p) => [...p, { game: +match[1], score }].slice(-600));
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
    setLiveMean(null);
    liveMeanRef.current = { sum: 0, count: 0 };
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
        {isAnyRunning && (
          <span className="badge badge--running">● Running</span>
        )}
      </div>
      <p className="panel-desc">
        Launch a Pygame window to watch models play, or play manually yourself. Each mode opens a separate window.
      </p>

      {/* ── Mode Cards ── */}
      <div className="watch-grid">
        {MODES.map((m) => (
          <div key={m.id} className={`watch-card watch-card--${m.color}`}>
            <span className="watch-icon">{m.icon}</span>
            <h3>{m.title}</h3>
            <p>{m.desc}</p>
            {m.needsCheckpoint && (
              <div className="form-group" style={{ marginTop: 'auto', marginBottom: '4px' }}>
                <select
                  className="input"
                  value={checkpoint}
                  onChange={(e) => setCheckpoint(e.target.value)}
                  disabled={isAnyRunning}
                  style={{ fontSize: 12, padding: "6px 10px" }}
                >
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

      {/* ── Stats + Chart area (only when we have data) ── */}
      {(liveData.length > 0 || history) && (
        <div className="watch-bottom-layout">
          <div className="watch-chart-section card">
            <div className="card-title">📈 Live Score Per Game</div>
            {liveData.length > 0 ? (
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={liveData} margin={{ top: 10, right: 20, left: -20, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(63,63,70,0.3)" />
                  <XAxis dataKey="game" stroke="#64748b" tick={{ fontSize: 11 }} />
                  <YAxis stroke="#64748b" tick={{ fontSize: 11 }} domain={[0, 253]} />
                  <Tooltip {...CHART_STYLE} />
                  <Line type="monotone" dataKey="score" stroke="#10b981" strokeWidth={2} dot={true} name="Score" />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <EmptyState icon="📊" title="No live data" desc="Launch a mode to see live scores." />
            )}
          </div>
          
          <div className="watch-stats-grid">
            <div className="watch-stat-card">
              <div className="watch-stat-label">Games Trained</div>
              <div className="watch-stat-value watch-stat-value--white">
                {history ? formatInteger(history.n_games) : "--"}
              </div>
            </div>
            <div className="watch-stat-card">
              <div className="watch-stat-label">Total Steps</div>
              <div className="watch-stat-value watch-stat-value--orange">
                {history ? formatInteger(history.total_steps) : "--"}
              </div>
            </div>
            <div className="watch-stat-card">
              <div className="watch-stat-label">Created</div>
              <div className="watch-stat-value watch-stat-value--blue" style={{ fontSize: "18px" }}>
                {history && history.creation_date
                  ? new Date(history.creation_date * 1000).toLocaleDateString("ru-RU", { day: "2-digit", month: "2-digit", year: "numeric" })
                  : "--"}
              </div>
            </div>
            <div className="watch-stat-card">
              <div className="watch-stat-label">Live Mean</div>
              <div className="watch-stat-value watch-stat-value--green">
                {liveMean !== null ? formatNumber(liveMean, 1) : "--"}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── Log ── */}
      {log.length > 0 && (
        <div className={`log-card${logCollapsed ? ' log-card--collapsed' : ''}`} style={{ marginTop: "20px", height: "240px" }}>
          <div className="log-head" onClick={() => setLogCollapsed(!logCollapsed)}>
            <div className="log-head-left">
              <span className="log-head-toggle">▼</span>
              <span className="log-head-title">Output</span>
              <span className="log-head-count">{log.length} lines</span>
            </div>
            <button className="btn-link" onClick={(e) => { e.stopPropagation(); setLog([]); setLiveData([]); setLiveMean(null); liveMeanRef.current = { sum: 0, count: 0 }; isUserScrolling.current = false; }}>Clear</button>
          </div>
          {!logCollapsed && (
            <div className="log-body" ref={logBodyRef} onScroll={handleScroll}>
              {log.map((l, i) => <div key={i} className="log-line"><span className="log-line-icon">›</span>{l}</div>)}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
