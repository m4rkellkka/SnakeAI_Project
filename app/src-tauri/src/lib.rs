use std::collections::HashMap;
use std::io::{BufRead, BufReader};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use tauri::{AppHandle, Emitter, State};

struct ProcessRegistry(Arc<Mutex<HashMap<String, u32>>>);

#[derive(serde::Serialize)]
struct EnvCheck {
    python_ok: bool,
    python_version: String,
    deps_ok: bool,
    missing_packages: Vec<String>,
}

#[tauri::command]
fn check_environment() -> EnvCheck {
    let python = find_python();

    let version_out = Command::new(&python).arg("--version").output();
    let (python_ok, python_version) = match version_out {
        Ok(out) if out.status.success() => {
            let v = String::from_utf8_lossy(&out.stdout).trim().to_string();
            let v = if v.is_empty() {
                String::from_utf8_lossy(&out.stderr).trim().to_string()
            } else {
                v
            };
            (true, v)
        }
        _ => (false, String::new()),
    };

    if !python_ok {
        return EnvCheck {
            python_ok: false,
            python_version: String::new(),
            deps_ok: false,
            missing_packages: vec!["torch".into(), "pygame".into(), "numpy".into(), "matplotlib".into()],
        };
    }

    // Check all packages in a single subprocess using find_spec —
    // avoids actually importing torch/CUDA which adds ~3s per call
    let check_script =
        "import importlib.util; \
         pkgs=['torch','pygame','numpy','matplotlib']; \
         missing=[p for p in pkgs if not importlib.util.find_spec(p)]; \
         print(','.join(missing),end='')";

    let result = Command::new(&python)
        .args(["-c", check_script])
        .output()
        .ok();

    let missing: Vec<String> = match result {
        Some(out) => {
            let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if s.is_empty() { vec![] } else { s.split(',').map(String::from).collect() }
        }
        None => vec!["torch".into(), "pygame".into(), "numpy".into(), "matplotlib".into()],
    };

    EnvCheck {
        python_ok: true,
        python_version,
        deps_ok: missing.is_empty(),
        missing_packages: missing,
    }
}

#[tauri::command]
fn install_deps(
    app: AppHandle,
    state: State<'_, ProcessRegistry>,
    cwd: String,
) -> Result<(), String> {
    let python = find_python();
    let id = "setup-install".to_string();

    {
        let map = state.0.lock().unwrap();
        if let Some(&pid) = map.get(&id) {
            kill_pid(pid);
        }
    }

    // Upgrade pip first, then install requirements — chained in one shell call
    let cmd = format!(
        "{py} -m pip install --upgrade pip && {py} -m pip install -r requirements.txt",
        py = python
    );

    let mut child = Command::new("/bin/sh")
        .args(["-c", &cmd])
        .current_dir(&cwd)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("Failed to start install: {}", e))?;

    let pid = child.id();
    let stdout = child.stdout.take();
    let stderr = child.stderr.take();

    {
        let mut map = state.0.lock().unwrap();
        map.insert(id.clone(), pid);
    }

    let registry = state.0.clone();

    if let Some(out) = stdout {
        let app_c = app.clone();
        let id_c = id.clone();
        std::thread::spawn(move || {
            for line in BufReader::new(out).lines().flatten() {
                let _ = app_c.emit(&format!("proc-out:{}", id_c), &line);
            }
        });
    }

    if let Some(err) = stderr {
        let app_c = app.clone();
        let id_c = id.clone();
        std::thread::spawn(move || {
            for line in BufReader::new(err).lines().flatten() {
                let _ = app_c.emit(&format!("proc-out:{}", id_c), &format!("[err] {}", line));
            }
        });
    }

    let app_wait = app.clone();
    let id_wait = id.clone();
    std::thread::spawn(move || {
        let _ = child.wait();
        {
            let mut map = registry.lock().unwrap();
            map.remove(&id_wait);
        }
        let _ = app_wait.emit(&format!("proc-done:{}", id_wait), ());
    });

    Ok(())
}

#[tauri::command]
fn open_url(url: String) {
    #[cfg(target_os = "macos")]
    { Command::new("open").arg(&url).spawn().ok(); }
    #[cfg(target_os = "windows")]
    { Command::new("cmd").args(["/c", "start", "", &url]).spawn().ok(); }
    #[cfg(target_os = "linux")]
    { Command::new("xdg-open").arg(&url).spawn().ok(); }
}

fn find_python() -> String {
    Command::new("which")
        .arg("python3")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "python3".to_string())
}

fn kill_pid(pid: u32) {
    Command::new("kill")
        .args(["-TERM", &pid.to_string()])
        .output()
        .ok();
}

#[tauri::command]
fn get_project_root() -> String {
    let cwd = std::env::current_dir().unwrap_or_default();
    // Walk up until we find src/train_ai.py
    let mut dir = cwd.clone();
    loop {
        if dir.join("src").join("train_ai.py").exists() {
            return dir.to_string_lossy().to_string();
        }
        match dir.parent() {
            Some(p) => dir = p.to_path_buf(),
            None => break,
        }
    }
    cwd.to_string_lossy().to_string()
}

#[tauri::command]
fn start_process(
    app: AppHandle,
    state: State<'_, ProcessRegistry>,
    id: String,
    args: Vec<String>,
    cwd: String,
) -> Result<(), String> {
    // Kill any existing process with this id first
    {
        let map = state.0.lock().unwrap();
        if let Some(&pid) = map.get(&id) {
            kill_pid(pid);
        }
    }

    let python = find_python();

    let mut child = Command::new(&python)
        .args(&args)
        .current_dir(&cwd)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("Failed to spawn python3: {}", e))?;

    let pid = child.id();
    let stdout = child.stdout.take();
    let stderr = child.stderr.take();

    {
        let mut map = state.0.lock().unwrap();
        map.insert(id.clone(), pid);
    }

    let registry = state.0.clone();

    if let Some(out) = stdout {
        let app_clone = app.clone();
        let id_clone = id.clone();
        std::thread::spawn(move || {
            for line in BufReader::new(out).lines().flatten() {
                let _ = app_clone.emit(&format!("proc-out:{}", id_clone), &line);
            }
        });
    }

    if let Some(err) = stderr {
        let app_clone = app.clone();
        let id_clone = id.clone();
        std::thread::spawn(move || {
            for line in BufReader::new(err).lines().flatten() {
                let _ = app_clone.emit(&format!("proc-out:{}", id_clone), &format!("[err] {}", line));
            }
        });
    }

    let app_wait = app.clone();
    let id_wait = id.clone();
    std::thread::spawn(move || {
        let _ = child.wait();
        {
            let mut map = registry.lock().unwrap();
            map.remove(&id_wait);
        }
        let _ = app_wait.emit(&format!("proc-done:{}", id_wait), ());
    });

    Ok(())
}

#[tauri::command]
fn stop_process(state: State<'_, ProcessRegistry>, id: String) -> Result<(), String> {
    let mut map = state.0.lock().unwrap();
    if let Some(pid) = map.remove(&id) {
        kill_pid(pid);
    }
    Ok(())
}

#[tauri::command]
fn list_processes(state: State<'_, ProcessRegistry>) -> Vec<String> {
    state.0.lock().unwrap().keys().cloned().collect()
}

#[tauri::command]
fn quit_app(app: AppHandle, state: State<'_, ProcessRegistry>) {
    let pids: Vec<u32> = state.0.lock().unwrap().values().cloned().collect();
    for pid in pids {
        kill_pid(pid);
    }
    app.exit(0);
}

#[tauri::command]
fn fetch_history(cwd: String, checkpoint_path: String) -> Result<String, String> {
    let python = find_python();
    let output = Command::new(&python)
        .arg("src/export_history.py")
        .arg(&checkpoint_path)
        .current_dir(&cwd)
        .output()
        .map_err(|e| format!("Failed to run export_history.py: {}", e))?;

    if !output.status.success() {
        let err = String::from_utf8_lossy(&output.stderr);
        return Err(format!("Python error: {}", err));
    }

    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

fn walk_checkpoints(dir: &std::path::Path, base: &std::path::Path, checkpoints: &mut Vec<String>) {
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                walk_checkpoints(&path, base, checkpoints);
            } else if path.extension().map_or(false, |ext| ext == "pth") {
                if let Ok(rel) = path.strip_prefix(base) {
                    checkpoints.push(rel.to_string_lossy().into_owned());
                }
            }
        }
    }
}

#[tauri::command]
fn list_checkpoints(cwd: String) -> Vec<String> {
    let mut checkpoints = Vec::new();
    let model_dir = std::path::PathBuf::from(&cwd).join("model");
    if model_dir.exists() {
        walk_checkpoints(&model_dir, &model_dir, &mut checkpoints);
    }
    checkpoints.sort();
    checkpoints
}

#[tauri::command]
fn delete_model(cwd: String, path: String) -> Result<(), String> {
    // Basic security check to prevent escaping the model directory
    if path.contains("..") || path.starts_with('/') {
        return Err("Invalid path".into());
    }
    
    let target = std::path::PathBuf::from(&cwd).join("model").join(&path);
    if !target.exists() {
        return Ok(());
    }
    
    // If it's the root checkpoint_best.pth or checkpoint_last.pth, we just delete the file
    // If it's a subfolder, we delete the whole directory
    if target.is_dir() {
        std::fs::remove_dir_all(&target).map_err(|e| format!("Failed to delete directory: {}", e))?;
    } else {
        // Find the parent directory to delete siblings if we are deleting "checkpoint_last.pth" from a subfolder
        // Actually, the user passes the path of the model (e.g. "my_model" or "checkpoint_last.pth")
        // But the ModelsPanel gives us the full relative path like "my_model/checkpoint_last.pth"
        let parent = target.parent();
        if let Some(p) = parent {
            if p.file_name().and_then(|n| n.to_str()) != Some("model") {
                // It's inside a subfolder, so delete the whole subfolder
                std::fs::remove_dir_all(p).map_err(|e| format!("Failed to delete subfolder: {}", e))?;
                return Ok(());
            }
        }
        // Otherwise, it's just a file in the root model folder
        std::fs::remove_file(&target).map_err(|e| format!("Failed to delete file: {}", e))?;
    }
    
    Ok(())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .manage(ProcessRegistry(Arc::new(Mutex::new(HashMap::new()))))
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![
            get_project_root,
            start_process,
            stop_process,
            list_processes,
            quit_app,
            fetch_history,
            list_checkpoints,
            delete_model,
            check_environment,
            install_deps,
            open_url,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application")
}
