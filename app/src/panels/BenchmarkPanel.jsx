import { useState, useEffect, useRef } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { formatCheckpoint } from "../utils";

const RUNNING_RE = /^Running \d+ games for ([^\s]+)/;
const STATS1_RE = /^\s*n=(\d+)\s+mean=([\d.]+)\s+median=([\d.]+)\s+std=([\d.]+)/;
const STATS2_RE = /^\s*min=([\d.]+)\s+p10=([\d.]+)\s+p25=([\d.]+)\s+p75=([\d.]+)\s+p90=([\d.]+)\s+max=([\d.]+)/;
const STATS3_RE = /^\s*win_rate=([\d.]+)%\s+stuck_rate=([\d.]+)%/;

export default function BenchmarkPanel({ projectRoot, isRunning, isActive }) {
  const [checkpoint, setCheckpoint] = useState("checkpoint_best.pth");
  const [compareCheckpoint, setCompareCheckpoint] = useState("none");
  const [checkpointList, setCheckpointList] = useState(["checkpoint_best.pth"]);
  const [games, setGames]           = useState("100");
  const [seed, setSeed]             = useState("42");
  const [noUnstick, setNoUnstick]   = useState(false);
  const [csvPath, setCsvPath]       = useState("");
  const [plotPath, setPlotPath]     = useState("");
  
  const [log, setLog]               = useState([]);
  const [results, setResults]       = useState([]);
  
  const logBodyRef = useRef(null);
  const isUserScrolling = useRef(false);
  const lastCheckpointRef = useRef(null);

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
    const u1 = listen("proc-out:benchmark", (e) => {
      const line = e.payload;
      setLog((p) => [...p, line].slice(-500));
      
      const mRun = RUNNING_RE.exec(line);
      if (mRun) {
        lastCheckpointRef.current = mRun[1];
        setResults(prev => [...prev, { checkpoint: mRun[1], complete: false }]);
      }
      
      const m1 = STATS1_RE.exec(line);
      if (m1) {
        setResults(prev => {
           const next = [...prev];
           if (next.length > 0) {
               next[next.length-1].n = m1[1];
               next[next.length-1].mean = m1[2];
               next[next.length-1].median = m1[3];
               next[next.length-1].std = m1[4];
           }
           return next;
        });
      }
      
      const m2 = STATS2_RE.exec(line);
      if (m2) {
        setResults(prev => {
           const next = [...prev];
           if (next.length > 0) {
               next[next.length-1].min = m2[1];
               next[next.length-1].max = m2[6];
           }
           return next;
        });
      }
      
      const m3 = STATS3_RE.exec(line);
      if (m3) {
        setResults(prev => {
           const next = [...prev];
           if (next.length > 0) {
               next[next.length-1].win_rate = m3[1];
               next[next.length-1].stuck_rate = m3[2];
               next[next.length-1].complete = true;
           }
           return next;
        });
      }
    });
    
    const u2 = listen("proc-done:benchmark", () => {});
    return () => { u1.then((f) => f()); u2.then((f) => f()); };
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

  const run = async () => {
    if (!projectRoot) return;
    const args = ["-u", "tools/benchmark.py", "--checkpoint", checkpoint, "--games", games, "--seed", seed];
    if (compareCheckpoint !== "none") {
      args.push("--checkpoint", compareCheckpoint);
    }
    if (noUnstick) args.push("--no-unstick");
    if (csvPath) { args.push("--csv", csvPath); }
    if (plotPath) { args.push("--plot", plotPath); }
    
    setLog([]);
    setResults([]);
    lastCheckpointRef.current = null;
    isUserScrolling.current = false;
    
    await invoke("start_process", { id: "benchmark", args, cwd: projectRoot });
  };

  const stop = async () => {
    await invoke("stop_process", { id: "benchmark" });
  };

  const lineClass = (l) => {
    if (l.includes("Mean") || l.includes("Median") || l.includes("Win rate")) return "log-line log-line--eval";
    if (l.includes("New best") || l.includes("===")) return "log-line log-line--best";
    return "log-line";
  };

  const completedResults = results.filter(r => r.complete);

  return (
    <div className="panel">
      <div className="panel-header">
        <h2>Benchmark</h2>
        <span className={`badge ${isRunning ? "badge--running" : "badge--idle"}`}>
          {isRunning ? "● Running" : "○ Idle"}
        </span>
      </div>
      <p className="panel-desc">
        Run N games and get full score statistics: mean, median, std, win rate, stuck rate.
      </p>

      <div className="form-grid">
        <div className="form-group">
          <span className="form-label">Checkpoint</span>
          <select className="input select" value={checkpoint} onChange={(e) => setCheckpoint(e.target.value)} disabled={isRunning}>
            {checkpointList.map(cp => <option key={cp} value={cp}>{formatCheckpoint(cp)}</option>)}
          </select>
        </div>
        <div className="form-group">
          <span className="form-label">Compare with (Optional)</span>
          <select className="input select" value={compareCheckpoint} onChange={(e) => setCompareCheckpoint(e.target.value)} disabled={isRunning}>
            <option value="none">-- None --</option>
            {checkpointList.map(cp => <option key={cp} value={cp}>{formatCheckpoint(cp)}</option>)}
          </select>
        </div>
        <div className="form-group">
          <span className="form-label">Games</span>
          <input className="input" type="number" min="1" value={games} onChange={(e) => setGames(e.target.value)} disabled={isRunning} />
        </div>
        <div className="form-group">
          <span className="form-label">Seed</span>
          <input className="input" type="number" value={seed} onChange={(e) => setSeed(e.target.value)} disabled={isRunning} />
        </div>
        <div className="form-group">
          <span className="form-label">Save CSV to</span>
          <input className="input" value={csvPath} onChange={(e) => setCsvPath(e.target.value)} placeholder="optional (e.g. data.csv)" disabled={isRunning} />
        </div>
        <div className="form-group">
          <span className="form-label">Save Plot to</span>
          <input className="input" value={plotPath} onChange={(e) => setPlotPath(e.target.value)} placeholder="optional (e.g. plot.png)" disabled={isRunning} />
        </div>
        <div className="form-group form-group--check">
          <label>
            <input type="checkbox" checked={noUnstick} onChange={(e) => setNoUnstick(e.target.checked)} disabled={isRunning} />
            No Unstick (raw network)
          </label>
        </div>
      </div>

      <div className="controls">
        <button className="btn btn--primary" onClick={run} disabled={isRunning || !projectRoot}>
          ▶ Run Benchmark
        </button>
        <button className="btn btn--stop" onClick={stop} disabled={!isRunning}>
          ■ Stop
        </button>
      </div>

      {completedResults.length > 0 && (
        <div className="results-container" style={{ marginTop: "20px", marginBottom: "20px" }}>
          <div className="benchmark-cards" style={{ display: "flex", gap: "16px", flexWrap: "wrap" }}>
            {completedResults.map((r, i) => (
              <div key={i} className="card" style={{ flex: 1, minWidth: "280px" }}>
                <div className="card-title" style={{ color: "#fff", marginBottom: "16px" }}>
                  {formatCheckpoint(r.checkpoint)}
                </div>
                <div className="stats">
                  <div className="stat">
                    <span className="stat-label">Mean Score</span>
                    <span className="stat-val stat-val--blue">{r.mean}</span>
                  </div>
                  <div className="stat">
                    <span className="stat-label">Median Score</span>
                    <span className="stat-val">{r.median}</span>
                  </div>
                  <div className="stat">
                    <span className="stat-label">Max Score</span>
                    <span className="stat-val">{r.max}</span>
                  </div>
                  <div className="stat">
                    <span className="stat-label">Win Rate</span>
                    <span className="stat-val stat-val--orange">{r.win_rate}%</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {log.length > 0 && (
        <div className="log-card">
          <div className="log-head">
            <span className="log-head-title">Output Log</span>
            <button className="btn-link" onClick={() => { setLog([]); isUserScrolling.current = false; }}>Clear</button>
          </div>
          <div className="log-body" ref={logBodyRef} onScroll={handleScroll}>
            {log.map((l, i) => <div key={i} className={lineClass(l)}>{l}</div>)}
          </div>
        </div>
      )}
    </div>
  );
}
