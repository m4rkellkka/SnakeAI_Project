import { useState, useEffect, useRef, useCallback } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";
import EmptyState from "../components/EmptyState";
import { formatCheckpoint, formatNumber, formatInteger } from "../utils";
import { usePersistedState, readPersisted, writePersisted, removePersisted } from "../hooks/usePersistedState";

const GAME_RE = /^Game: (\d+)(?:\s*\[.*?\])?\s*\|\s*Score: (\d+)\s*\|\s*Loss: ([\d.]+)\s*\|\s*Steps: (\d+)(?:\s*\|\s*Reward:\s*([\-\d.]+))?/;
const EVAL_RE = /Honest eval: avg=([\d.]+), max=(\d+)/;

const CHART_STYLE = {
  contentStyle: { background: "rgba(12,12,18,0.95)", border: "1px solid rgba(63,63,70,0.4)", borderRadius: 10, fontSize: 12, backdropFilter: "blur(12px)" },
  labelStyle: { color: "#94a3b8" },
};

const lineIcon = (line) => {
  if (line.includes("New best")) return "🏆";
  if (line.includes("Honest eval")) return "🎯";
  if (line.includes("[C")) return "📚";
  if (line.includes("[D")) return "🔀";
  return "›";
};

export default function TrainPanel({ projectRoot, isRunning, isActive }) {
  // Log + stat cards are seeded from the last persisted snapshot so they
  // reappear instantly after a reload (charts are rebuilt from disk instead —
  // see loadHistory / activeRun below).
  const [log, setLog]           = useState(() => readPersisted("train:log", []));
  const [gameData, setGameData] = useState([]);
  const [evalData, setEvalData] = useState([]);
  const [lossData, setLossData] = useState([]);
  const [rewardData, setRewardData] = useState([]);
  const [lastGame, setLastGame] = useState(() => readPersisted("train:lastGame", null));
  const [lastEval, setLastEval] = useState(() => readPersisted("train:lastEval", null));
  const [logCollapsed, setLogCollapsed] = usePersistedState("train:logCollapsed", false);
  const [runAlgo, setRunAlgo] = useState("bc");

  const [checkpointList, setCheckpointList] = useState(["checkpoint_last.pth"]);
  const [loadCheckpoint, setLoadCheckpoint] = usePersistedState("train:loadCheckpoint", "checkpoint_last.pth");
  // The checkpoint path the *currently training* run writes to. While set (i.e.
  // a run is live), history is always loaded from here so a reload rebuilds the
  // correct run's graphs — not the dropdown's stale default. Cleared on stop.
  const [activeRun, setActiveRun] = useState(() => readPersisted("train:activeRun", null));

  const logBuf      = useRef([]);
  const gameBuf     = useRef([]);
  const lossBuf     = useRef([]);
  const evalBuf     = useRef([]);
  const rewardBuf   = useRef([]);
  const lossMeanRef = useRef({ sum: 0, count: 0 });
  const logBodyRef  = useRef(null);
  const lastGameRef = useRef(null);
  const isUserScrolling = useRef(false);
  // Latest snapshot of the state worth persisting, read by the throttled
  // persister / beforeunload handler without re-subscribing every render.
  const snapshotRef = useRef({ log, lastGame, lastEval });

  useEffect(() => {
    if (projectRoot && isActive) {
      invoke("list_checkpoints", { cwd: projectRoot }).then((list) => {
        if (list.length > 0) {
          const filtered = list.filter(cp => cp.endsWith("checkpoint_last.pth"));
          setCheckpointList(["none", ...filtered]);
          if (!filtered.includes(loadCheckpoint) && loadCheckpoint !== "none") {
            setLoadCheckpoint(filtered.includes("checkpoint_last.pth") ? "checkpoint_last.pth" : "none");
          }
        }
      });
    }
  }, [projectRoot]);

  const loadHistory = useCallback(async () => {
    if (!projectRoot) return;
    try {
      // A live run's path (activeRun) always wins over the dropdown selection,
      // so after a reload we rebuild the run that's actually training.
      const checkpointPath = activeRun
        ? activeRun
        : (loadCheckpoint === "none" ? "model/checkpoint_last.pth" : `model/${loadCheckpoint}`);
      const histStr = await invoke("fetch_history", { cwd: projectRoot, checkpointPath });
      const hist = JSON.parse(histStr);
      if (hist.error) return;
      
      const newLossData = [];
      const newEvalData = [];
      const newGameData = [];
      const newRewardData = [];
      
      if (hist.algo) {
        setRunAlgo(hist.algo);
      } else {
        setRunAlgo("bc");
      }
      
      if (hist.loss_history && hist.mean_loss_history) {
        for (let i = 0; i < hist.loss_history.length; i++) {
          newLossData.push({
            game: i + 1,
            loss: hist.loss_history[i],
            meanLoss: hist.mean_loss_history[i]
          });
        }
      }
      if (hist.eval_games_history) {
        for (let i = 0; i < hist.eval_games_history.length; i++) {
          const game = hist.eval_games_history[i];
          const closestLoss = newLossData.find(l => l.game >= game)?.loss || 0;
          newEvalData.push({
            game: game,
            avg: hist.eval_avg_history[i],
            max: hist.eval_max_history[i],
            loss: closestLoss
          });
        }
      }
      
      let start_game_idx = Math.max(1, (hist.n_games || 0) - (hist.score_history?.length || 0) + 1);
      if (hist.score_history) {
        for (let i = 0; i < hist.score_history.length; i++) {
          newGameData.push({
            game: start_game_idx + i,
            score: hist.score_history[i]
          });
        }
      }
      
      if (hist.reward_history) {
        let r_start = Math.max(1, (hist.n_games || 0) - (hist.reward_history.length || 0) + 1);
        for (let i = 0; i < hist.reward_history.length; i++) {
          newRewardData.push({
            game: r_start + i,
            reward: hist.reward_history[i]
          });
        }
      }
      
      // Seed the running-mean accumulator so live loss points continue the
      // mean from where the loaded history left off (instead of restarting at 0).
      lossMeanRef.current = {
        sum: newLossData.reduce((s, l) => s + l.loss, 0),
        count: newLossData.length,
      };

      setLossData(newLossData.slice(-600));
      setEvalData(newEvalData.slice(-300));
      setGameData(newGameData.slice(-600));
      setRewardData(newRewardData.slice(-600));
      
      if (newEvalData.length > 0) {
        setLastEval(newEvalData[newEvalData.length - 1]);
      }
      if (newLossData.length > 0) {
        const lastL = newLossData[newLossData.length - 1];
        const lastScore = newGameData.length > 0 ? newGameData[newGameData.length - 1].score : 0;
        setLastGame({ game: lastL.game, score: lastScore, loss: lastL.loss, steps: hist.total_steps || 0 });
      }
    } catch (e) {
      console.error("Failed to load history:", e);
    }
  }, [projectRoot, loadCheckpoint, activeRun]);

  // Load on mount and whenever the source checkpoint changes. `isRunning` is
  // intentionally NOT a dependency: a status flip mid-session must not reload
  // from (stale) disk and clobber newer streamed points.
  useEffect(() => {
    loadHistory();
  }, [loadHistory]);

  useEffect(() => {
    const unOut  = listen("proc-out:train", (e) => {
      const line = e.payload;
      logBuf.current.push(line);

      const gm = GAME_RE.exec(line);
      if (gm) {
        const gameIdx = +gm[1];
        const loss = +parseFloat(gm[3]).toFixed(4);
        const steps = +gm[4];
        const rewardStr = gm[5];
        const reward = rewardStr !== undefined ? +parseFloat(rewardStr).toFixed(2) : undefined;
        
        const entry = { game: gameIdx, score: +gm[2], loss, steps, reward };
        gameBuf.current.push(entry);
        lossBuf.current.push({ game: gameIdx, loss });
        if (reward !== undefined) {
          rewardBuf.current.push({ game: gameIdx, reward });
        }
        lastGameRef.current = entry;
      }

      const em = EVAL_RE.exec(line);
      if (em) {
        const game = lastGameRef.current?.game ?? 0;
        const loss = lastGameRef.current?.loss ?? 0;
        const entry = { game, avg: +parseFloat(em[1]).toFixed(1), max: +em[2], loss };
        evalBuf.current.push(entry);
        setLastEval(entry);
      }
    });

    const unDone = listen("proc-done:train", () => {
      logBuf.current.push("── process finished ──");
      // Run is over — hand control back to the dropdown for previewing history.
      setActiveRun(null);
      removePersisted("train:activeRun");
    });

    const flush = setInterval(() => {
      if (logBuf.current.length) {
        const next = logBuf.current.splice(0);
        setLog((p) => [...p, ...next].slice(-1000));
      }
      if (gameBuf.current.length) {
        const next = gameBuf.current.splice(0);
        setLastGame(next[next.length - 1]);
        setGameData((p) => [...p, ...next].slice(-600));
      }
      if (lossBuf.current.length) {
        const raw = lossBuf.current.splice(0);
        const next = raw.map(({ game, loss }) => {
          lossMeanRef.current.sum += loss;
          lossMeanRef.current.count += 1;
          return { game, loss, meanLoss: lossMeanRef.current.sum / lossMeanRef.current.count };
        });
        setLossData((p) => [...p, ...next].slice(-600));
      }
      if (evalBuf.current.length) {
        const next = evalBuf.current.splice(0);
        setEvalData((p) => [...p, ...next].slice(-300));
      }
      if (rewardBuf.current.length) {
        const next = rewardBuf.current.splice(0);
        setRewardData((p) => [...p, ...next].slice(-600));
      }
    }, 120);

    return () => {
      unOut.then((f) => f());
      unDone.then((f) => f());
      clearInterval(flush);
    };
  }, []);

  useEffect(() => {
    if (logBodyRef.current && !isUserScrolling.current) {
      logBodyRef.current.scrollTop = logBodyRef.current.scrollHeight;
    }
  }, [log]);

  // Keep the snapshot ref current, then throttle-persist it (every 2s) plus a
  // final flush on beforeunload — so the log + stat cards survive a reload.
  useEffect(() => {
    snapshotRef.current = { log, lastGame, lastEval };
  }, [log, lastGame, lastEval]);

  useEffect(() => {
    const persist = () => {
      const s = snapshotRef.current;
      writePersisted("train:log", s.log.slice(-300));
      writePersisted("train:lastGame", s.lastGame);
      writePersisted("train:lastEval", s.lastEval);
    };
    const id = setInterval(persist, 2000);
    window.addEventListener("beforeunload", persist);
    return () => {
      clearInterval(id);
      window.removeEventListener("beforeunload", persist);
    };
  }, []);

  const handleScroll = () => {
    if (!logBodyRef.current) return;
    const { scrollTop, scrollHeight, clientHeight } = logBodyRef.current;
    isUserScrolling.current = scrollHeight - scrollTop - clientHeight > 10;
  };

  const handleStart = async () => {
    if (!projectRoot) return;
    setLog([]); setGameData([]); setEvalData([]); setLossData([]); setRewardData([]);
    setLastGame(null); setLastEval(null);
    lastGameRef.current = null;
    logBuf.current = []; gameBuf.current = []; lossBuf.current = []; evalBuf.current = []; rewardBuf.current = [];
    lossMeanRef.current = { sum: 0, count: 0 };
    isUserScrolling.current = false;
    
    const args = ["-u", runAlgo === "dqn" ? "src/train_rl.py" : "src/train_ai.py"];

    // Default run writes to model/checkpoint_last.pth; a named run writes to
    // model/<run-name>/checkpoint_last.pth. Remember whichever it is so a reload
    // reloads the right history (the dropdown is locked while training).
    let activeCheckpointPath = "model/checkpoint_last.pth";

    if (loadCheckpoint === "none") {
      args.push("--load-checkpoint", "none");
    } else if (loadCheckpoint !== "checkpoint_last.pth") {
      const parts = loadCheckpoint.split("/");
      if (parts.length > 1) {
        args.push("--run-name", parts[0]);
        args.push("--load-checkpoint", parts[1]);
        activeCheckpointPath = `model/${parts[0]}/checkpoint_last.pth`;
      } else {
        args.push("--load-checkpoint", parts[0]);
      }
    }

    setActiveRun(activeCheckpointPath);
    writePersisted("train:activeRun", activeCheckpointPath);

    await invoke("start_process", { id: "train", args, cwd: projectRoot });
  };

  const handleStop = async () => {
    await invoke("stop_process", { id: "train" });
    setActiveRun(null);
    removePersisted("train:activeRun");
  };

  const lineClass = (line) => {
    if (line.includes("New best")) return "log-line log-line--best";
    if (line.includes("Honest eval")) return "log-line log-line--eval";
    if (line.includes("[C") || line.includes("[D")) return "log-line log-line--tag";
    return "log-line";
  };

  const hasData = lastGame || lastEval || gameData.length > 0;

  return (
    <div className="panel">
      <div className="panel-header">
        <h2>Training</h2>
        <span className={`badge ${isRunning ? "badge--running" : "badge--idle"}`}>
          {isRunning ? "● Running" : "○ Idle"}
        </span>
      </div>
      <p className="panel-desc">
        {runAlgo === "dqn" 
          ? "Train a neural network to play Snake using Reinforcement Learning (Double DQN)."
          : "Train a neural network to play Snake using behavioral cloning (DAgger) and curriculum strategies."}
      </p>

      <div className="controls">
        <button className="btn btn--primary" onClick={handleStart} disabled={isRunning || !projectRoot}>
          ▶ Start Training
        </button>
        <button className="btn btn--stop" onClick={handleStop} disabled={!isRunning}>
          ■ Stop
        </button>
        
        <div className="form-group form-group--inline" style={{ marginLeft: "8px" }}>
          <span className="form-label">Resume from:</span>
          <select 
            className="input" 
            value={loadCheckpoint} 
            onChange={(e) => setLoadCheckpoint(e.target.value)} 
            disabled={isRunning}
            style={{ width: "240px" }}
          >
            {checkpointList.map(cp => <option key={cp} value={cp}>{formatCheckpoint(cp)}</option>)}
          </select>
        </div>
      </div>

      {/* ── Stats Row ── */}
      {hasData && (
        <div className="stats">
          <div className="stat stat--blue">
            <span className="stat-label">Game</span>
            <span className="stat-val stat-val--blue">{lastGame?.game ?? "—"}</span>
            <span className="stat-hint">Episodes played</span>
          </div>
          <div className="stat stat--purple">
            <span className="stat-label">Loss</span>
            <span className="stat-val stat-val--purple">{lastGame?.loss?.toFixed(4) ?? "—"}</span>
            <span className="stat-hint">Cross-entropy</span>
          </div>
          {runAlgo === "dqn" && (
            <div className="stat stat--yellow">
              <span className="stat-label">Reward</span>
              <span className="stat-val stat-val--yellow">{lastGame?.reward?.toFixed(1) ?? "—"}</span>
              <span className="stat-hint">Episode total</span>
            </div>
          )}
          <div className="stat stat--green">
            <span className="stat-label">Eval Avg</span>
            <span className="stat-val stat-val--green">{lastEval ? formatNumber(lastEval.avg, 1) : "—"}</span>
            <span className="stat-hint">Honest evaluation</span>
          </div>
          <div className="stat stat--orange">
            <span className="stat-label">Steps</span>
            <span className="stat-val stat-val--orange">{lastGame ? formatInteger(lastGame.steps) : "—"}</span>
            <span className="stat-hint">Total env steps</span>
          </div>
          {lastEval && (
            <div className="stat stat--cyan">
              <span className="stat-label">Eval Max</span>
              <span className="stat-val stat-val--cyan">{lastEval.max}</span>
              <span className="stat-hint">Best single game</span>
            </div>
          )}
        </div>
      )}

      {!hasData && !isRunning && (
        <EmptyState
          icon="🧠"
          title="Ready to train"
          desc="Click 'Start Training' to begin teaching the AI to play Snake. Charts and metrics will appear here in real-time."
        />
      )}

      {/* ── Charts + Log Layout ── */}
      {hasData && (
        <div className="train-layout">
          <div className="charts-col">
            {evalData.length > 0 && (
              <div className="card">
                <div className="card-title">📊 Honest Eval & Loss</div>
                <ResponsiveContainer width="100%" height={220}>
                  <LineChart data={evalData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(63,63,70,0.3)" />
                    <XAxis dataKey="game" stroke="#64748b" tick={{ fontSize: 11 }} />
                    <YAxis yAxisId="left" domain={[0, 253]} stroke="#64748b" tick={{ fontSize: 11 }} />
                    <YAxis yAxisId="right" orientation="right" stroke="#64748b" tick={{ fontSize: 11 }} />
                    <Tooltip {...CHART_STYLE} formatter={(val, name) => [formatNumber(val, name.includes("Loss") ? 4 : 1), name]} />
                    <Legend wrapperStyle={{ fontSize: 12 }} />
                    <Line yAxisId="left" type="monotone" dataKey="avg" stroke="#8b5cf6" dot={true} strokeWidth={2} name="Eval Avg" />
                    <Line yAxisId="left" type="monotone" dataKey="max" stroke="#10b981" dot={true} strokeWidth={1.5} name="Eval Max" />
                    <Line yAxisId="right" type="monotone" dataKey="loss" stroke="#ef4444" dot={true} strokeWidth={1.5} name="Loss at Eval" opacity={0.7} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            )}

            <div className="card">
              <div className="card-title">📉 Loss & Mean Loss</div>
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={lossData.slice(-600)}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(63,63,70,0.3)" />
                  <XAxis dataKey="game" stroke="#64748b" tick={{ fontSize: 11 }} />
                  <YAxis stroke="#64748b" tick={{ fontSize: 11 }} />
                  <Tooltip {...CHART_STYLE} formatter={(val, name) => [formatNumber(val, 4), name]} />
                  <Legend wrapperStyle={{ fontSize: 12 }} />
                  <Line type="monotone" dataKey="loss" stroke="#ef4444" dot={false} strokeWidth={1.5} name="Loss" opacity={0.5} />
                  <Line type="monotone" dataKey="meanLoss" stroke="#f59e0b" dot={false} strokeWidth={2} name="Mean Loss" />
                </LineChart>
              </ResponsiveContainer>
            </div>

            <div className="card">
              <div className="card-title">🍎 Score Per Game</div>
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={gameData.slice(-600)}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(63,63,70,0.3)" />
                  <XAxis dataKey="game" stroke="#64748b" tick={{ fontSize: 11 }} />
                  <YAxis domain={[0, 253]} stroke="#64748b" tick={{ fontSize: 11 }} />
                  <Tooltip {...CHART_STYLE} />
                  <Line type="monotone" dataKey="score" stroke="#10b981" dot={false} strokeWidth={1.5} name="Score" />
                </LineChart>
              </ResponsiveContainer>
            </div>

            {runAlgo === "dqn" && rewardData.length > 0 && (
              <div className="card">
                <div className="card-title">💰 Reward Per Game</div>
                <ResponsiveContainer width="100%" height={200}>
                  <LineChart data={rewardData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(63,63,70,0.3)" />
                    <XAxis dataKey="game" stroke="#64748b" tick={{ fontSize: 11 }} />
                    <YAxis stroke="#64748b" tick={{ fontSize: 11 }} />
                    <Tooltip {...CHART_STYLE} formatter={(val, name) => [formatNumber(val, 2), name]} />
                    <Line type="monotone" dataKey="reward" stroke="#eab308" dot={false} strokeWidth={1.5} name="Reward" />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            )}
          </div>

          {/* ── Collapsible Log ── */}
          <div className={`log-card${logCollapsed ? ' log-card--collapsed' : ''}`}>
            <div className="log-head" onClick={() => setLogCollapsed(!logCollapsed)}>
              <div className="log-head-left">
                <span className="log-head-toggle">▼</span>
                <span className="log-head-title">Output Log</span>
                <span className="log-head-count">{log.length} lines</span>
              </div>
              <button className="btn-link" onClick={(e) => { e.stopPropagation(); setLog([]); isUserScrolling.current = false; }}>Clear</button>
            </div>
            {!logCollapsed && (
              <div className="log-body" ref={logBodyRef} onScroll={handleScroll}>
                {log.length === 0
                  ? <span className="log-empty">Start training to see output…</span>
                  : log.map((l, i) => (
                    <div key={i} className={lineClass(l)}>
                      <span className="log-line-icon">{lineIcon(l)}</span>
                      {l}
                    </div>
                  ))}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
