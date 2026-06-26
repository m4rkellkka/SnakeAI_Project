use std::collections::HashMap;
use std::io::{BufRead, BufReader};
use std::os::unix::process::CommandExt;
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use tauri::{AppHandle, Emitter, State};

struct ProcessRegistry(Arc<Mutex<HashMap<String, u32>>>);

#[derive(serde::Serialize)]
struct EnvCheck {
    python_ok: bool,
    python_version: String,
    python_path: String,
    project_root: String,
    deps_ok: bool,
    missing_packages: Vec<String>,
}

#[tauri::command]
fn check_environment() -> EnvCheck {
    let base = get_project_root();
    let python = find_python(&base);

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
            python_path: python.clone(),
            project_root: base.clone(),
            deps_ok: false,
            missing_packages: vec!["torch".into(), "pygame".into(), "numpy".into(), "matplotlib".into()],
        };
    }

    // Use `pip show` — checks the same package database pip installs into,
    // more reliable than find_spec which depends on sys.path quirks.
    let pkgs = ["torch", "pygame", "numpy", "matplotlib"];
    let show_result = Command::new(&python)
        .args(
            std::iter::once("-m")
                .chain(std::iter::once("pip"))
                .chain(std::iter::once("show"))
                .chain(pkgs.iter().copied())
                .collect::<Vec<_>>(),
        )
        .output()
        .ok();

    let missing: Vec<String> = match show_result {
        Some(out) => {
            let stdout = String::from_utf8_lossy(&out.stdout).to_lowercase();
            pkgs.iter()
                .filter(|&&p| !stdout.contains(&format!("name: {}", p)))
                .map(|&p| p.to_string())
                .collect()
        }
        None => pkgs.iter().map(|&s| s.to_string()).collect(),
    };

    EnvCheck {
        python_ok: true,
        python_version,
        python_path: python.clone(),
        project_root: base.clone(),
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
    let python = find_python(&cwd);
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
        let status = child.wait().ok();
        let success = status.map(|s| s.success()).unwrap_or(false);
        {
            let mut map = registry.lock().unwrap();
            map.remove(&id_wait);
        }
        let _ = app_wait.emit(&format!("proc-done:{}", id_wait), success);
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

fn find_python(base: &str) -> String {
    // Prefer a project-local venv — guarantees the right packages regardless of system state
    let venv = std::path::Path::new(base).join(".venv").join("bin").join("python3");
    if venv.exists() {
        return venv.to_string_lossy().to_string();
    }

    // Fallback: use a login shell so PATH matches the user's terminal (homebrew, pyenv, etc.)
    let via_shell = Command::new("/bin/zsh")
        .args(["-l", "-c", "which python3"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());

    if let Some(p) = via_shell {
        return p;
    }

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
    // Kill the whole process group so child processes (python, torch) die too.
    Command::new("kill")
        .args(["-TERM", &format!("-{}", pid)])
        .output()
        .ok();
    Command::new("kill")
        .args(["-TERM", &pid.to_string()])
        .output()
        .ok();
    let pid_copy = pid;
    std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_millis(300));
        Command::new("kill")
            .args(["-KILL", &format!("-{}", pid_copy)])
            .output()
            .ok();
        Command::new("kill")
            .args(["-KILL", &pid_copy.to_string()])
            .output()
            .ok();
    });
}

fn is_valid_project_root(path: &str) -> bool {
    std::path::Path::new(path).join("src").join("train_ai.py").exists()
}

fn config_file_path() -> std::path::PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
    std::path::Path::new(&home).join(".snake_ai_project_root")
}

#[tauri::command]
fn get_project_root() -> String {
    // 1. Check saved config file
    if let Ok(saved) = std::fs::read_to_string(config_file_path()) {
        let saved = saved.trim().to_string();
        if is_valid_project_root(&saved) {
            return saved;
        }
    }

    // 2. Walk up from cwd
    let cwd = std::env::current_dir().unwrap_or_default();
    let mut dir = cwd.clone();
    loop {
        if is_valid_project_root(&dir.to_string_lossy()) {
            return dir.to_string_lossy().to_string();
        }
        match dir.parent() {
            Some(p) => dir = p.to_path_buf(),
            None => break,
        }
    }

    // 3. Try common locations
    let home = std::env::var("HOME").unwrap_or_default();
    let candidates = [
        format!("{}/Documents/GitHub/SnakeAI_Project", home),
        format!("{}/SnakeAI_Project", home),
        format!("{}/Desktop/SnakeAI_Project", home),
        format!("{}/Downloads/SnakeAI_Project", home),
        format!("{}/Projects/SnakeAI_Project", home),
    ];
    for candidate in &candidates {
        if is_valid_project_root(candidate) {
            return candidate.clone();
        }
    }

    // Not found — return empty string so the UI can ask the user
    String::new()
}

#[tauri::command]
fn set_project_root(path: String) -> Result<String, String> {
    let path = path.trim().to_string();
    if !is_valid_project_root(&path) {
        return Err(
            "src/train_ai.py not found in that folder. Make sure this is the SnakeAI_Project directory.".to_string()
        );
    }
    std::fs::write(config_file_path(), &path).map_err(|e| e.to_string())?;
    Ok(path)
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

    let python = find_python(&cwd);

    let mut child = Command::new(&python)
        .args(&args)
        .current_dir(&cwd)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .process_group(0)
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
    let expected_pid = pid;
    std::thread::spawn(move || {
        let _ = child.wait();
        {
            let mut map = registry.lock().unwrap();
            if map.get(&id_wait) == Some(&expected_pid) {
                map.remove(&id_wait);
            }
        }
        let _ = app_wait.emit(&format!("proc-done:{}", id_wait), expected_pid);
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
    let python = find_python(&cwd);
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
            set_project_root,
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
