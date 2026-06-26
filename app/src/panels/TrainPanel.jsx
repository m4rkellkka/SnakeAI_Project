import { useState, useEffect, useRef, useCallback } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";
import EmptyState from "../components/EmptyState";
import AnimatedNumber from "../components/AnimatedNumber";
import { formatCheckpoint, formatNumber, formatInteger } from "../utils";
import { usePersistedState, readPersisted, writePersisted, removePersisted } from "../hooks/usePersistedState";

const GAME_RE = /^Game: (\d+)(?:\s*\[.*?\])?\s*\|\s*Score: (\d+)\s*\|\s*Loss: ([\d.]+)\s*\|\s*Steps: (\d+)(?:\s*\|\s*Reward:\s*([\-\d.]+))?/;
const EVAL_RE = /Honest eval: avg=([\d.]+), max=(\d+)/;

const CHART_STYLE = {
  contentStyle: { background: "rgba(12,12,18,0.95)", border: "1px solid rgba(63,63,70,0.4)", borderRadius: 10, fontSize: 12, backdropFilter: "blur(12px)" },
  labelStyle: { color: "#94a3b8" },
};

const lineIcon = (line) => {
  if (line.includes("New best")) return "\u{1F3C6}";
  if (line.includes("Honest eval")) return "\u{1F3AF}";
  if (line.includes("[C")) return "\u{1F4DA}";
  if (line.includes("[D")) return "\u{1F500}";
  return "›";
};

export default function TrainPanel({ projectRoot, isRunning, isActive }) {
  // ── Model sidebar ──
  const [checkpointList, setCheckpointList] = useState([]);
  const [selectedModel, setSelectedModel] = usePersistedState("train:selectedModel", "checkpoint_last.pth");
  const [historyLoading, setHistoryLoading] = useState(true);

  // ── Training data ──
  const [log, setLog]           = useState(() => readPersisted("train:log", []));
  const [gameData, setGameData] = useState([]);
  const [evalData, setEvalData] = useState([]);
  const [lossData, setLossData] = useState([]);
  const [rewardData, setRewardData] = useState([]);
  const [lastGame, setLastGame] = useState(() => readPersisted("train:lastGame", null));
  const [lastEval, setLastEval] = useState(() => readPersisted("train:lastEval", null));
  const [logCollapsed, setLogCollapsed] = usePersistedState("train:logCollapsed", false);
  const [runAlgo, setRunAlgo]     = useState("bc");
  const [runConfig, setRunConfig] = useState({});
  const [activeRun, setActiveRun] = useState(() => readPersisted("train:activeRun", null));
  const [statsKey, setStatsKey]   = useState(0);

  const logBuf      = useRef([]);
  const gameBuf     = useRef([]);
  const lossBuf     = useRef([]);
  const evalBuf     = useRef([]);
  const rewardBuf   = useRef([]);
  const lossMeanRef = useRef({ sum: 0, count: 0 });
  const logBodyRef  = useRef(null);
  const lastGameRef = useRef(null);
  const isUserScrolling = useRef(false);
  const snapshotRef = useRef({ log, lastGame, lastEval });

  // ── Load model list ──
  useEffect(() => {
    if (projectRoot && isActive) {
      invoke("list_checkpoints", { cwd: projectRoot }).then((list) => {
        const filtered = list.filter(cp => cp.endsWith("checkpoint_last.pth"));
        setCheckpointList(filtered);
        if (selectedModel !== "none" && !filtered.includes(selectedModel)) {
          setSelectedModel(filtered.length > 0 ? filtered[0] : "none");
        }
      });
    }
  }, [projectRoot, isActive]);

  // ── Load history for the selected model ──
  const loadHistory = useCallback(async () => {
    if (!projectRoot) return;
    setHistoryLoading(true);
    try {
      const checkpointPath = activeRun
        ? activeRun
        : (selectedModel === "none" ? null : `model/${selectedModel}`);

      if (!checkpointPath) { setHistoryLoading(false); return; }

      const histStr = await invoke("fetch_history", { cwd: projectRoot, checkpointPath });
      const hist = JSON.parse(histStr);
      if (hist.error) { setHistoryLoading(false); return; }

      const newLossData = [];
      const newEvalData = [];
      const newGameData = [];
      const newRewardData = [];

      setRunAlgo(hist.algo || "bc");
      setRunConfig(hist.run_config || {});

      if (hist.loss_history) {
        const hasMean = hist.mean_loss_history?.length === hist.loss_history.length;
        let runSum = 0;
        for (let i = 0; i < hist.loss_history.length; i++) {
          const loss = hist.loss_history[i];
          runSum += loss;
          newLossData.push({
            game: i + 1, loss,
            meanLoss: hasMean ? hist.mean_loss_history[i] : runSum / (i + 1),
          });
        }
      }
      if (hist.eval_games_history) {
        for (let i = 0; i < hist.eval_games_history.length; i++) {
          const game = hist.eval_games_history[i];
          const closestLoss = newLossData.find(l => l.game >= game)?.loss || 0;
          newEvalData.push({ game, avg: hist.eval_avg_history[i], max: hist.eval_max_history[i], loss: closestLoss });
        }
      }
      let startIdx = Math.max(1, (hist.n_games || 0) - (hist.score_history?.length || 0) + 1);
      if (hist.score_history) {
        for (let i = 0; i < hist.score_history.length; i++) {
          newGameData.push({ game: startIdx + i, score: hist.score_history[i] });
        }
      }
      if (hist.reward_history) {
        let rStart = Math.max(1, (hist.n_games || 0) - (hist.reward_history.length || 0) + 1);
        for (let i = 0; i < hist.reward_history.length; i++) {
          newRewardData.push({ game: rStart + i, reward: hist.reward_history[i] });
        }
      }

      lossMeanRef.current = { sum: newLossData.reduce((s, l) => s + l.loss, 0), count: newLossData.length };

      setLossData(newLossData.slice(-600));
      setEvalData(newEvalData.slice(-300));
      setGameData(newGameData.slice(-600));
      setRewardData(newRewardData.slice(-600));
      if (newEvalData.length > 0) setLastEval(newEvalData[newEvalData.length - 1]);
      if (newLossData.length > 0) {
        const lastL = newLossData[newLossData.length - 1];
        const lastScore = newGameData.length > 0 ? newGameData[newGameData.length - 1].score : 0;
        setLastGame({ game: lastL.game, score: lastScore, loss: lastL.loss, steps: hist.total_steps || 0 });
      }
    } catch (e) {
      console.error("Failed to load history:", e);
    }
    setHistoryLoading(false);
  }, [projectRoot, selectedModel, activeRun]);

  useEffect(() => { loadHistory(); }, [loadHistory]);

  // ── Event streaming (train process) ──
  useEffect(() => {
    const unOut = listen("proc-out:train", (e) => {
      const line = e.payload;
      logBuf.current.push(line);
      const gm = GAME_RE.exec(line);
      if (gm) {
        const gameIdx = +gm[1], loss = +parseFloat(gm[3]).toFixed(4), steps = +gm[4];
        const rewardStr = gm[5];
        const reward = rewardStr !== undefined ? +parseFloat(rewardStr).toFixed(2) : undefined;
        const entry = { game: gameIdx, score: +gm[2], loss, steps, reward };
        gameBuf.current.push(entry);
        lossBuf.current.push({ game: gameIdx, loss });
        if (reward !== undefined) rewardBuf.current.push({ game: gameIdx, reward });
        lastGameRef.current = entry;
      }
      const em = EVAL_RE.exec(line);
      if (em) {
        const game = lastGameRef.current?.game ?? 0, loss = lastGameRef.current?.loss ?? 0;
        const entry = { game, avg: +parseFloat(em[1]).toFixed(1), max: +em[2], loss };
        evalBuf.current.push(entry);
        setLastEval(entry);
      }
    });
    const unDone = listen("proc-done:train", () => {
      logBuf.current.push("── process finished ──");
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
    return () => { unOut.then(f => f()); unDone.then(f => f()); clearInterval(flush); };
  }, []);

  // ── Auto-scroll log ──
  useEffect(() => {
    if (logBodyRef.current && !isUserScrolling.current)
      logBodyRef.current.scrollTop = logBodyRef.current.scrollHeight;
  }, [log]);

  // ── Persist snapshots ──
  useEffect(() => { snapshotRef.current = { log, lastGame, lastEval }; }, [log, lastGame, lastEval]);
  useEffect(() => {
    const persist = () => {
      const s = snapshotRef.current;
      writePersisted("train:log", s.log.slice(-300));
      writePersisted("train:lastGame", s.lastGame);
      writePersisted("train:lastEval", s.lastEval);
    };
    const id = setInterval(persist, 2000);
    window.addEventListener("beforeunload", persist);
    return () => { clearInterval(id); window.removeEventListener("beforeunload", persist); };
  }, []);

  const handleScroll = () => {
    if (!logBodyRef.current) return;
    const { scrollTop, scrollHeight, clientHeight } = logBodyRef.current;
    isUserScrolling.current = scrollHeight - scrollTop - clientHeight > 10;
  };

  // ── Shared reset helper ──
  const resetData = () => {
    setLog([]); setGameData([]); setEvalData([]); setLossData([]); setRewardData([]);
    setLastGame(null); setLastEval(null);
    logBuf.current = []; gameBuf.current = []; lossBuf.current = [];
    evalBuf.current = []; rewardBuf.current = [];
    lossMeanRef.current = { sum: 0, count: 0 };
    lastGameRef.current = null;
    isUserScrolling.current = false;
  };

  // ── Model selection (destroys old tiles, creates fresh ones) ──
  const handleModelSelect = (model) => {
    if (isRunning || model === selectedModel) return;
    setHistoryLoading(true);
    resetData();
    setRunAlgo("bc");
    setRunConfig({});
    setActiveRun(null);
    removePersisted("train:activeRun");
    setSelectedModel(model);
    setStatsKey(k => k + 1);
  };

  // ── Start / Stop ──
  const handleStart = async () => {
    if (!projectRoot) return;
    resetData();
    setStatsKey(k => k + 1);

    const isRL = runAlgo === "dqn";
    const args = ["-u", isRL ? "src/train_rl.py" : "src/train_ai.py"];
    if (isRL) args.push("--algo", "dqn");

    let activeCheckpointPath = "model/checkpoint_last.pth";
    if (selectedModel === "none") {
      args.push("--load-checkpoint", "none");
    } else if (selectedModel !== "checkpoint_last.pth") {
      const parts = selectedModel.split("/");
      if (parts.length > 1) {
        args.push("--run-name", parts[0]);
        args.push("--load-checkpoint", parts[1]);
        activeCheckpointPath = `model/${parts[0]}/checkpoint_last.pth`;
      } else {
        args.push("--load-checkpoint", parts[0]);
      }
    }

    if (isRL) {
      if (runConfig?.lr != null) args.push("--lr", String(runConfig.lr));
      if (runConfig?.gamma != null) args.push("--gamma", String(runConfig.gamma));
      args.push("--warmstart-episodes", String(runConfig?.warmstart_episodes ?? 50));
      if (runConfig?.teacher_bias === false) args.push("--no-teacher-bias");
    } else {
      if (runConfig?.lr != null) args.push("--lr", String(runConfig.lr));
      if (runConfig?.dagger_prob_max != null) args.push("--dagger-prob-max", String(runConfig.dagger_prob_max));
      if (runConfig?.curriculum_prob != null) args.push("--curriculum-prob", String(runConfig.curriculum_prob));
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

  const runChips = runAlgo === "dqn"
    ? [
        { label: "method", value: "Double DQN", hint: "Self-learning from a reward signal." },
        { label: "lr", value: runConfig?.lr ?? "—", hint: "Adam learning rate." },
        { label: "γ", value: runConfig?.gamma ?? "—", hint: "Discount factor." },
        { label: "warm-start", value: `${runConfig?.warmstart_episodes ?? 50} ep`, hint: "Teacher pre-fill episodes." },
        { label: "teacher bias", value: runConfig?.teacher_bias === false ? "off" : "on", hint: "Teacher-guided early exploration." },
      ]
    : [
        { label: "method", value: "Behavioral Cloning", hint: "Imitates the Hamiltonian-cycle teacher." },
        { label: "lr", value: runConfig?.lr ?? "—", hint: "Adam learning rate." },
        { label: "DAgger max", value: runConfig?.dagger_prob_max ?? "—", hint: "Max network-driven step probability." },
        { label: "curriculum", value: runConfig?.curriculum_prob ?? "—", hint: "Late-game start probability." },
      ];

  return (
    <div className="panel">
      <div className="panel-header">
        <h2>Training</h2>
        <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
          <span className={`algo-badge algo-badge--${runAlgo === "dqn" ? "rl" : "bc"}`}>
            {runAlgo === "dqn" ? "RL · DQN" : "BC"}
          </span>
          <span className={`badge ${isRunning ? "badge--running" : "badge--idle"}`}>
            {isRunning ? "● Running" : "○ Idle"}
          </span>
        </div>
      </div>
      <p className="panel-desc">
        Select a model from the list, then start training. Statistics update in real-time once training begins.
      </p>

      <div className="train-two-col">
        {/* Left: Model List */}
        <aside className="train-sidebar">
          <div className="train-sidebar-head">Models</div>
          <div className="train-sidebar-list">
            <button
              className={`train-model-item${selectedModel === "none" ? " train-model-item--active" : ""}`}
              onClick={() => handleModelSelect("none")}
              disabled={isRunning}
            >
              <span className="train-model-dot train-model-dot--new" />
              <div className="train-model-info">
                <span className="train-model-name">From Scratch</span>
                <span className="train-model-meta">Random weights</span>
              </div>
            </button>

            {checkpointList.map(cp => (
              <button
                key={cp}
                className={`train-model-item${selectedModel === cp ? " train-model-item--active" : ""}`}
                onClick={() => handleModelSelect(cp)}
                disabled={isRunning}
              >
                <span className="train-model-dot" />
                <div className="train-model-info">
                  <span className="train-model-name">{formatCheckpoint(cp)}</span>
                  <span className="train-model-meta">{cp}</span>
                </div>
              </button>
            ))}

            {checkpointList.length === 0 && (
              <div className="train-sidebar-empty">No models yet. Create one in the Models tab.</div>
            )}
          </div>
        </aside>

        {/* Right: Training Content */}
        <div className="train-main">
          <div className="controls">
            <button className="btn btn--primary" onClick={handleStart} disabled={isRunning || !projectRoot}>
              ▶ Start Training
            </button>
            <button className="btn btn--stop" onClick={handleStop} disabled={!isRunning}>
              ■ Stop
            </button>
          </div>

          <div className="run-chips">
            {runChips.map(c => (
              <span key={c.label} className="run-chip" title={c.hint}>
                <span className="run-chip-key">{c.label}</span>
                <span className="run-chip-val">{c.value}</span>
              </span>
            ))}
          </div>

          {/* Stats & Charts — keyed so React fully destroys and recreates
              the DOM subtree on model switch, giving clean entrance animations
              and preventing stale-tile flicker. */}
          <div key={statsKey}>
            {hasData ? (
              <>
                <div className="stats">
                  <div className="stat stat--blue">
                    <span className="stat-label">Game</span>
                    <AnimatedNumber className="stat-val stat-val--blue" value={lastGame?.game} format={n => Math.round(n).toString()} />
                    <span className="stat-hint">Episodes played</span>
                  </div>
                  <div className="stat stat--purple">
                    <span className="stat-label">Loss</span>
                    <AnimatedNumber className="stat-val stat-val--purple" value={lastGame?.loss} format={n => n.toFixed(4)} />
                    <span className="stat-hint">{runAlgo === "dqn" ? "TD error (Huber)" : "Cross-entropy"}</span>
                  </div>
                  {runAlgo === "dqn" && (
                    <div className="stat stat--yellow">
                      <span className="stat-label">Reward</span>
                      <AnimatedNumber className="stat-val stat-val--yellow" value={lastGame?.reward} format={n => n.toFixed(1)} />
                      <span className="stat-hint">Episode total</span>
                    </div>
                  )}
                  <div className="stat stat--green">
                    <span className="stat-label">Eval Avg</span>
                    <AnimatedNumber className="stat-val stat-val--green" value={lastEval?.avg} format={n => formatNumber(n, 1)} />
                    <span className="stat-hint">Honest evaluation</span>
                  </div>
                  <div className="stat stat--orange">
                    <span className="stat-label">Steps</span>
                    <AnimatedNumber className="stat-val stat-val--orange" value={lastGame?.steps} format={n => formatInteger(Math.round(n))} />
                    <span className="stat-hint">Total env steps</span>
                  </div>
                  {lastEval && (
                    <div className="stat stat--cyan">
                      <span className="stat-label">Eval Max</span>
                      <AnimatedNumber className="stat-val stat-val--cyan" value={lastEval.max} format={n => Math.round(n).toString()} />
                      <span className="stat-hint">Best single game</span>
                    </div>
                  )}
                </div>

                <div className="train-charts-log">
                  <div className="charts-col">
                    {evalData.length > 0 && (
                      <div className="card">
                        <div className="card-title">Honest Eval & Loss</div>
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
                      <div className="card-title">{runAlgo === "dqn" ? "TD Loss (Huber)" : "Loss & Mean Loss"}</div>
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
                      <div className="card-title">Score Per Game</div>
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
                        <div className="card-title">Reward Per Game</div>
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

                  <div className={`log-card${logCollapsed ? " log-card--collapsed" : ""}`}>
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
              </>
            ) : historyLoading ? (
              <div className="train-loading">
                <div className="train-loading-spinner" />
                <span>Loading model data…</span>
              </div>
            ) : !isRunning ? (
              <EmptyState
                icon="\u{1F9E0}"
                title="Ready to train"
                desc="Click 'Start Training' to begin teaching the AI to play Snake. Charts and metrics will appear here in real-time."
              />
            ) : null}
          </div>
        </div>
      </div>
    </div>
  );
}
