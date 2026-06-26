import { useState, useEffect, useRef, useCallback } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import GameCanvas from "../components/GameCanvas";
import { usePersistedState } from "../hooks/usePersistedState";
import {
  createState, step, actionForDesiredDir, STRAIGHT, GRID,
} from "../game/engine";

// ── Manual play: speed presets → tick interval (ms). ──
const SPEEDS = [
  { label: "Chill", ms: 170 },
  { label: "Normal", ms: 120 },
  { label: "Fast", ms: 85 },
  { label: "Insane", ms: 55 },
];

// ── Watch AI: speed presets → python step delay (s). tickMs follows so the
// canvas interpolation window matches the stream cadence. ──
const AI_SPEEDS = [
  { label: "Slow", delay: 0.12 },
  { label: "Normal", delay: 0.07 },
  { label: "Fast", delay: 0.035 },
  { label: "Turbo", delay: 0.0 },
];

const AI_MODES = [
  { id: "pretrained", label: "Pretrained" },
  { id: "checkpoint", label: "Checkpoint" },
  { id: "teacher", label: "Teacher" },
];

const PROC_ID = "play-watch";

const KEY_DIR = {
  ArrowUp: "UP", KeyW: "UP",
  ArrowDown: "DOWN", KeyS: "DOWN",
  ArrowLeft: "LEFT", KeyA: "LEFT",
  ArrowRight: "RIGHT", KeyD: "RIGHT",
};

function parseFrame(line) {
  if (!line || line[0] !== "{") return null;
  try {
    const j = JSON.parse(line);
    if (!Array.isArray(j.snake)) return null;
    return {
      snake: j.snake.map(([x, y]) => ({ x, y })),
      foods: (j.foods || []).map(([x, y]) => ({ x, y })),
      dir: j.dir,
      score: j.score,
      over: j.over,
      game: j.game,
    };
  } catch {
    return null;
  }
}

function buildAiArgs(modeId, checkpointPath) {
  if (modeId === "teacher") return ["--mode", "teacher"];
  if (modeId === "pretrained") return ["--mode", "network", "--checkpoint", "pretrained.pth"];
  return ["--mode", "network", "--checkpoint", checkpointPath || "checkpoint_best.pth"];
}

export default function PlayPanel({ isActive, projectRoot, runningList }) {
  const [mode, setMode] = usePersistedState("play:mode", "you");

  // ── Manual state ──
  const [frame, setFrame] = useState(() => createState(1));
  const [playing, setPlaying] = useState(false);
  const [started, setStarted] = useState(false);
  const [speedIdx, setSpeedIdx] = usePersistedState("play:speed", 1);
  const [best, setBest] = usePersistedState("play:best", 0);

  // ── AI state ──
  const [aiFrame, setAiFrame] = useState(null);
  const [aiMode, setAiMode] = usePersistedState("play:aimode", "pretrained");
  const [aiCheckpoint, setAiCheckpoint] = usePersistedState("play:aickpt", "checkpoint_best.pth");
  const [aiSpeedIdx, setAiSpeedIdx] = usePersistedState("play:aispeed", 1);
  const [checkpoints, setCheckpoints] = useState([]);

  const [aiWants, setAiWants] = useState(false);
  const aiWantsRef = useRef(false);
  const [aiLog, setAiLog] = useState([]);
  const lastGameRef = useRef(0);

  const actionRef = useRef(STRAIGHT);
  const frameRef = useRef(frame);
  frameRef.current = frame;

  const tickMs = SPEEDS[speedIdx]?.ms ?? 120;
  const aiDelay = AI_SPEEDS[aiSpeedIdx]?.delay ?? 0.07;
  const aiTickMs = Math.max(35, aiDelay * 1000);
  const over = frame.over;
  const aiRunning = !!runningList?.includes(PROC_ID);

  // ── Load checkpoints list ──
  useEffect(() => {
    if (!projectRoot) return;
    invoke("list_checkpoints", { cwd: projectRoot }).then((list) => {
      const filtered = list.filter((p) => p !== "pretrained.pth");
      setCheckpoints(filtered);
    });
  }, [projectRoot, isActive]);

  // ─────────────────────────── Manual play ───────────────────────────
  const restart = useCallback(() => {
    actionRef.current = STRAIGHT;
    setFrame(createState(1));
    setStarted(true);
    setPlaying(true);
  }, []);

  useEffect(() => {
    if (mode !== "you" || !playing) return;
    const id = setInterval(() => {
      const next = step(frameRef.current, actionRef.current);
      actionRef.current = STRAIGHT;
      setFrame(next);
      if (next.over) {
        setPlaying(false);
        setBest((b) => Math.max(b, next.score));
      }
    }, tickMs);
    return () => clearInterval(id);
  }, [mode, playing, tickMs, setBest]);

  useEffect(() => {
    if (!isActive || mode !== "you") return;
    const onKey = (e) => {
      if (e.code === "Space") {
        e.preventDefault();
        if (over || !started) restart();
        else setPlaying((p) => !p);
        return;
      }
      const desired = KEY_DIR[e.code];
      if (!desired) return;
      e.preventDefault();
      if (!started || over) { restart(); return; }
      if (!playing) setPlaying(true);
      const action = actionForDesiredDir(frameRef.current.dir, desired);
      if (action !== null) actionRef.current = action;
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [isActive, mode, playing, over, started, restart]);

  // ─────────────────────────── Watch AI ───────────────────────────
  useEffect(() => {
    const un = listen(`proc-out:${PROC_ID}`, (e) => {
      if (!aiWantsRef.current) return;
      const f = parseFrame(e.payload);
      if (!f) return;
      setAiFrame(f);
      if (f.over && f.game !== lastGameRef.current) {
        lastGameRef.current = f.game;
        setAiLog((prev) => {
          const entry = { game: f.game, score: f.score, len: f.snake.length };
          const next = [entry, ...prev];
          return next.length > 50 ? next.slice(0, 50) : next;
        });
      }
    });
    return () => { un.then((f) => f()); };
  }, []);

  const launchAI = useCallback(async (modeId, ckpt, delay, keepFrame = false) => {
    if (!projectRoot) return;
    const srcArgs = buildAiArgs(modeId, ckpt);
    if (!keepFrame) { setAiFrame(null); setAiLog([]); lastGameRef.current = 0; }
    setAiWants(true);
    aiWantsRef.current = true;
    await invoke("start_process", {
      id: PROC_ID,
      args: ["-u", "src/stream_game.py", ...srcArgs, "--delay", String(delay)],
      cwd: projectRoot,
    });
  }, [projectRoot]);

  const startAI = useCallback(
    () => launchAI(aiMode, aiCheckpoint, aiDelay),
    [launchAI, aiMode, aiCheckpoint, aiDelay],
  );

  const stopAI = useCallback(async () => {
    aiWantsRef.current = false;
    setAiWants(false);
    setAiFrame(null);
    await invoke("stop_process", { id: PROC_ID });
  }, []);

  const pickAiSpeed = (i) => {
    setAiSpeedIdx(i);
    if (aiWants) launchAI(aiMode, aiCheckpoint, AI_SPEEDS[i].delay, true);
  };

  // Pause manual & stop AI when leaving the tab.
  useEffect(() => {
    if (!isActive) {
      setPlaying(false);
      if (aiRunning || aiWants) stopAI();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isActive]);

  // Switching modes: pause manual / stop AI so they never run at once.
  const switchMode = (m) => {
    if (m === mode) return;
    setPlaying(false);
    if (mode === "ai" && (aiRunning || aiWants)) stopAI();
    setMode(m);
  };

  const shownFrame = mode === "you" ? frame : aiFrame;
  const shownTickMs = mode === "you" ? tickMs : aiTickMs;
  const displayScore = shownFrame?.score ?? 0;
  const displayLen = shownFrame?.snake?.length ?? 0;
  const fill = Math.round((displayLen / (GRID * GRID)) * 100);

  return (
    <div className="panel">
      <div className="panel-header">
        <h2>Play</h2>
        {((mode === "you" && playing) || (mode === "ai" && aiWants)) && (
          <span className="badge badge--running">● {mode === "you" ? "Playing" : "Watching"}</span>
        )}
      </div>
      <p className="panel-desc">
        Play Snake yourself or watch the AI play — both rendered live in the same
        in-app board. Steer with <kbd>WASD</kbd> / arrows.
      </p>

      {/* ── Mode toggle ── */}
      <div className="play-mode-toggle">
        <button
          className={`play-mode-btn${mode === "you" ? " play-mode-btn--active" : ""}`}
          onClick={() => switchMode("you")}
        >🎮 You</button>
        <button
          className={`play-mode-btn${mode === "ai" ? " play-mode-btn--active" : ""}`}
          onClick={() => switchMode("ai")}
        >🤖 Watch AI</button>
      </div>

      <div className="play-layout">
        <div className="play-stage">
          <GameCanvas key={mode} frame={shownFrame} tickMs={shownTickMs} size={560} />

          {/* Manual overlays */}
          {mode === "you" && (!started || over || !playing) && (
            <div className="play-overlay">
              {over ? (
                <>
                  <div className="play-overlay-title">Game Over</div>
                  <div className="play-overlay-score">Score: {frame.score}</div>
                  <button className="btn btn--primary" onClick={restart}>↻ Play again</button>
                </>
              ) : !started ? (
                <>
                  <div className="play-overlay-title">🐍 Snake</div>
                  <div className="play-overlay-hint">Press any arrow / WASD to start</div>
                  <button className="btn btn--primary" onClick={restart}>▶ Start</button>
                </>
              ) : (
                <>
                  <div className="play-overlay-title">Paused</div>
                  <button className="btn btn--primary" onClick={() => setPlaying(true)}>▶ Resume</button>
                </>
              )}
            </div>
          )}

          {/* AI: loading overlay only while waiting for first frame */}
          {mode === "ai" && aiWants && !aiFrame && (
            <div className="play-overlay">
              <div className="play-overlay-hint">Loading model…</div>
            </div>
          )}
        </div>

        <div className="play-side">
          <div className="play-stat-card">
            <div className="play-stat-label">Score</div>
            <div className="play-stat-value play-stat-value--green">{displayScore}</div>
          </div>
          <div className="play-stat-row">
            <div className="play-stat-card play-stat-card--sm">
              <div className="play-stat-label">{mode === "you" ? "Best" : "Game"}</div>
              <div className="play-stat-value play-stat-value--orange">
                {mode === "you" ? best : (shownFrame?.game ?? "--")}
              </div>
            </div>
            <div className="play-stat-card play-stat-card--sm">
              <div className="play-stat-label">Length</div>
              <div className="play-stat-value play-stat-value--blue">{displayLen}</div>
            </div>
          </div>

          <div className="play-stat-card play-stat-card--sm">
            <div className="play-stat-label">Board filled</div>
            <div className="play-fill-bar">
              <div className="play-fill-bar-inner" style={{ width: `${fill}%` }} />
            </div>
            <div className="play-fill-pct">{fill}%</div>
          </div>

          {mode === "ai" && (
            <>
              <div className="play-controls">
                <div className="play-controls-label">Source</div>
                <div className="play-speed-group">
                  {AI_MODES.map((m) => (
                    <button
                      key={m.id}
                      className={`play-speed-btn${m.id === aiMode ? " play-speed-btn--active" : ""}`}
                      onClick={() => setAiMode(m.id)}
                      disabled={aiWants}
                    >{m.label}</button>
                  ))}
                </div>
              </div>

              {aiMode === "checkpoint" && (
                <div className="play-controls">
                  <div className="play-controls-label">Checkpoint</div>
                  <select
                    className="play-select"
                    value={aiCheckpoint}
                    onChange={(e) => setAiCheckpoint(e.target.value)}
                    disabled={aiWants}
                  >
                    {checkpoints.length === 0 && (
                      <option value="">No checkpoints found</option>
                    )}
                    {checkpoints.map((c) => (
                      <option key={c} value={c}>{c}</option>
                    ))}
                  </select>
                </div>
              )}
            </>
          )}

          <div className="play-controls">
            <div className="play-controls-label">Speed</div>
            <div className="play-speed-group">
              {(mode === "you" ? SPEEDS : AI_SPEEDS).map((s, i) => (
                <button
                  key={s.label}
                  className={`play-speed-btn${i === (mode === "you" ? speedIdx : aiSpeedIdx) ? " play-speed-btn--active" : ""}`}
                  onClick={() => (mode === "you" ? setSpeedIdx(i) : pickAiSpeed(i))}
                >{s.label}</button>
              ))}
            </div>
          </div>

          {mode === "ai" && aiLog.length > 0 && (
            <div className="play-log">
              <div className="play-controls-label">
                Results
                <span className="play-log-avg">
                  avg {(aiLog.reduce((s, e) => s + e.score, 0) / aiLog.length).toFixed(1)}
                </span>
              </div>
              <div className="play-log-list">
                {aiLog.map((e) => (
                  <div key={e.game} className="play-log-row">
                    <span className="play-log-game">#{e.game}</span>
                    <span className="play-log-score">{e.score}</span>
                    <span className="play-log-len">len {e.len}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          <div className="play-actions">
            {mode === "you" ? (
              <>
                {playing ? (
                  <button className="btn btn--ghost" onClick={() => setPlaying(false)}>⏸ Pause</button>
                ) : (
                  <button className="btn btn--ghost" onClick={() => (over || !started ? restart() : setPlaying(true))}>
                    {over || !started ? "▶ Start" : "▶ Resume"}
                  </button>
                )}
                <button className="btn btn--ghost" onClick={restart}>↻ Restart</button>
              </>
            ) : (
              aiWants ? (
                <button className="btn btn--stop" onClick={stopAI}>■ Stop</button>
              ) : (
                <button className="btn btn--ghost" onClick={startAI} disabled={!projectRoot || (aiMode === "checkpoint" && checkpoints.length === 0)}>
                  ▶ Start
                </button>
              )
            )}
          </div>

          {mode === "you" && (
            <div className="play-help">
              <div><kbd>W</kbd><kbd>A</kbd><kbd>S</kbd><kbd>D</kbd> / arrows — steer</div>
              <div><kbd>Space</kbd> — pause / resume</div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
