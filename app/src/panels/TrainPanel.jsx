import { useState, useEffect, useRef, useCallback } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";

import { formatCheckpoint, formatNumber, formatInteger } from "../utils";

const GAME_RE = /^Game: (\d+)(?:\s*\[.*?\])?\s*\|\s*Score: (\d+)\s*\|\s*Loss: ([\d.]+)\s*\|\s*Steps: (\d+)/;
const EVAL_RE = /Honest eval: avg=([\d.]+), max=(\d+)/;

const CHART_STYLE = {
  contentStyle: { background: "rgba(17,17,24,0.9)", border: "1px solid rgba(30,30,46,0.8)", borderRadius: 12, fontSize: 12, backdropFilter: "blur(10px)" },
  labelStyle: { color: "#8b8b9a" },
};

export default function TrainPanel({ projectRoot, isRunning, isActive }) {
  const [log, setLog]           = useState([]);
  const [gameData, setGameData] = useState([]);
  const [evalData, setEvalData] = useState([]);
  const [lossData, setLossData] = useState([]);
  const [lastGame, setLastGame] = useState(null);
  const [lastEval, setLastEval] = useState(null);
  
  const [checkpointList, setCheckpointList] = useState(["checkpoint_last.pth"]);
  const [loadCheckpoint, setLoadCheckpoint] = useState("checkpoint_last.pth");

  const logBuf      = useRef([]);
  const gameBuf     = useRef([]);
  const evalBuf     = useRef([]);
  const logBodyRef  = useRef(null);
  const lastGameRef = useRef(null);
  const isUserScrolling = useRef(false);

  useEffect(() => {
    if (projectRoot && isActive) {
      invoke("list_checkpoints", { cwd: projectRoot }).then((list) => {
        if (list.length > 0) {
          // Keep only 'checkpoint_last.pth' models for training
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
      const histStr = await invoke("fetch_history", { cwd: projectRoot, checkpointPath: loadCheckpoint === "none" ? "model/checkpoint_last.pth" : `model/${loadCheckpoint}` });
      const hist = JSON.parse(histStr);
      if (hist.error) return;
      
      const newLossData = [];
      const newEvalData = [];
      const newGameData = [];
      
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
      
      setLossData(newLossData);
      setEvalData(newEvalData);
      setGameData(newGameData);
      
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
  }, [projectRoot, loadCheckpoint]);

  useEffect(() => {
    loadHistory();
  }, [loadHistory, isRunning]);

  useEffect(() => {
    const unOut  = listen("proc-out:train", (e) => {
      const line = e.payload;
      logBuf.current.push(line);

      const gm = GAME_RE.exec(line);
      if (gm) {
        const gameIdx = +gm[1];
        const loss = +parseFloat(gm[3]).toFixed(4);
        const steps = +gm[4];
        const entry = { game: gameIdx, score: +gm[2], loss, steps };
        gameBuf.current.push(entry);
        lastGameRef.current = entry;
        
        setLossData(prev => {
          const meanLoss = prev.length > 0 ? (prev[prev.length-1].meanLoss * prev.length + loss) / (prev.length + 1) : loss;
          return [...prev, { game: gameIdx, loss, meanLoss }];
        });
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
      if (evalBuf.current.length) {
        const next = evalBuf.current.splice(0);
        setEvalData((p) => [...p, ...next]);
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

  const handleScroll = () => {
    if (!logBodyRef.current) return;
    const { scrollTop, scrollHeight, clientHeight } = logBodyRef.current;
    isUserScrolling.current = scrollHeight - scrollTop - clientHeight > 10;
  };

  const handleStart = async () => {
    if (!projectRoot) return;
    setLog([]); setGameData([]); setEvalData([]); setLossData([]);
    setLastGame(null); setLastEval(null);
    lastGameRef.current = null;
    isUserScrolling.current = false;
    
    const args = ["-u", "src/train_ai.py"];
    
    if (loadCheckpoint === "none") {
      args.push("--load-checkpoint", "none");
    } else if (loadCheckpoint !== "checkpoint_last.pth") {
      const parts = loadCheckpoint.split("/");
      if (parts.length > 1) {
        args.push("--run-name", parts[0]);
        args.push("--load-checkpoint", parts[1]);
      } else {
        args.push("--load-checkpoint", parts[0]);
      }
    }
    
    await invoke("start_process", { id: "train", args, cwd: projectRoot });
  };

  const handleStop = async () => {
    await invoke("stop_process", { id: "train" });
  };

  const lineClass = (line) => {
    if (line.includes("New best")) return "log-line log-line--best";
    if (line.includes("Honest eval")) return "log-line log-line--eval";
    if (line.includes("[C") || line.includes("[D")) return "log-line log-line--tag";
    return "log-line";
  };

  return (
    <div className="panel">
      <div className="panel-header">
        <h2>Training</h2>
        <span className={`badge ${isRunning ? "badge--running" : "badge--idle"}`}>
          {isRunning ? "● Running" : "○ Idle"}
        </span>
      </div>

      <div className="controls" style={{ alignItems: "center" }}>
        <button className="btn btn--primary" onClick={handleStart} disabled={isRunning || !projectRoot}>
          ▶ Start Training
        </button>
        <button className="btn btn--stop" onClick={handleStop} disabled={!isRunning}>
          ■ Stop
        </button>
        
        <div className="form-group" style={{ marginLeft: "12px", flexDirection: "row", alignItems: "center", gap: "12px" }}>
          <span className="form-label" style={{ margin: 0 }}>Resume from:</span>
          <select 
            className="input select" 
            value={loadCheckpoint} 
            onChange={(e) => setLoadCheckpoint(e.target.value)} 
            disabled={isRunning}
            style={{ width: "240px" }}
          >
            {checkpointList.map(cp => <option key={cp} value={cp}>{formatCheckpoint(cp)}</option>)}
          </select>
        </div>
      </div>

      {(lastGame || lastEval) && (
        <div className="stats">
          {lastGame && (
            <>
              <div className="stat">
                <span className="stat-label">Game</span>
                <span className="stat-val">{lastGame.game}</span>
              </div>
              <div className="stat">
                <span className="stat-label">Loss</span>
                <span className="stat-val">{lastGame.loss?.toFixed(4)}</span>
              </div>
            </>
          )}
          {lastEval && (
            <>
              <div className="stat">
                <span className="stat-label">Eval Avg</span>
                <span className="stat-val stat-val--blue">{formatNumber(lastEval.avg, 1)}</span>
              </div>
              <div className="stat">
                <span className="stat-label">Steps</span>
                <span className="stat-val stat-val--orange">{formatInteger(lastGame?.steps || 0)}</span>
              </div>
            </>
          )}
        </div>
      )}

      <div className="train-layout">
        <div className="charts-col">
          {evalData.length > 0 && (
            <div className="card">
              <div className="card-title">HONEST EVAL & LOSS</div>
              <ResponsiveContainer width="100%" height={140}>
                <LineChart data={evalData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(30,30,46,0.6)" />
                  <XAxis dataKey="game" stroke="#8b8b9a" tick={{ fontSize: 10 }} />
                  <YAxis yAxisId="left" domain={[0, 253]} stroke="#8b8b9a" tick={{ fontSize: 10 }} />
                  <YAxis yAxisId="right" orientation="right" stroke="#8b8b9a" tick={{ fontSize: 10 }} />
                  <Tooltip {...CHART_STYLE} formatter={(val, name) => [formatNumber(val, name.includes("Loss") ? 4 : 1), name]} />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  <Line yAxisId="left" type="monotone" dataKey="avg" stroke="#7c6af5" dot={true} strokeWidth={2} name="Eval Avg" />
                  <Line yAxisId="left" type="monotone" dataKey="max" stroke="#00ff87" dot={true} strokeWidth={1.5} name="Eval Max" />
                  <Line yAxisId="right" type="monotone" dataKey="loss" stroke="#ff5c5c" dot={true} strokeWidth={1.5} name="Loss at Eval" opacity={0.8} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}

          <div className="card chart-gap">
            <div className="card-title">LOSS & MEAN LOSS</div>
            <ResponsiveContainer width="100%" height={140}>
              <LineChart data={lossData.slice(-600)}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(30,30,46,0.6)" />
                <XAxis dataKey="game" stroke="#8b8b9a" tick={{ fontSize: 10 }} />
                <YAxis stroke="#8b8b9a" tick={{ fontSize: 10 }} />
                <Tooltip {...CHART_STYLE} formatter={(val, name) => [formatNumber(val, 4), name]} />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Line type="monotone" dataKey="loss" stroke="#ff5c5c" dot={false} strokeWidth={1.5} name="Loss" opacity={0.6} />
                <Line type="monotone" dataKey="meanLoss" stroke="#ff9f43" dot={false} strokeWidth={2} name="Mean Loss" />
              </LineChart>
            </ResponsiveContainer>
          </div>

          <div className="card chart-gap">
            <div className="card-title">SCORE PER GAME</div>
            <ResponsiveContainer width="100%" height={140}>
              <LineChart data={gameData.slice(-600)}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(30,30,46,0.6)" />
                <XAxis dataKey="game" stroke="#8b8b9a" tick={{ fontSize: 10 }} />
                <YAxis domain={[0, 253]} stroke="#8b8b9a" tick={{ fontSize: 10 }} />
                <Tooltip {...CHART_STYLE} />
                <Line type="monotone" dataKey="score" stroke="#00ff87" dot={false} strokeWidth={1.5} name="Score" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="log-card">
          <div className="log-head">
            <span className="log-head-title">Output Log</span>
            <button className="btn-link" onClick={() => { setLog([]); isUserScrolling.current = false; }}>Clear</button>
          </div>
          <div className="log-body" ref={logBodyRef} onScroll={handleScroll}>
            {log.length === 0
              ? <span className="log-empty">Start training to see output…</span>
              : log.map((l, i) => <div key={i} className={lineClass(l)}>{l}</div>)}
          </div>
        </div>
      </div>
    </div>
  );
}
