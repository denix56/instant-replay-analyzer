use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;

#[derive(Default)]
struct ProcessState {
    backend: Mutex<Option<Child>>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct NativeRuntimeConfig {
    clips_dir: Option<String>,
    runtime_profile: Option<String>,
    model_tier: Option<String>,
    backend_port: Option<u16>,
}

#[derive(Serialize)]
struct CommandResult {
    ok: bool,
    code: Option<i32>,
    stdout: String,
    stderr: String,
}

#[derive(Serialize)]
struct RuntimeProbe {
    uv_available: bool,
    uv_version: String,
    python_available: bool,
    python_version: String,
    ffmpeg_available: bool,
    ffmpeg_version: String,
    backend_running: bool,
    project_root: String,
    data_dir: String,
    models_dir: String,
    log_file: String,
}

fn project_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."))
}

fn home_dir() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
}

fn search_executable(names: &[&str], extra_candidates: &[PathBuf]) -> Option<PathBuf> {
    if let Some(path_var) = std::env::var_os("PATH") {
        for dir in std::env::split_paths(&path_var) {
            for name in names {
                let candidate = dir.join(name);
                if candidate.is_file() {
                    return Some(candidate);
                }
            }
        }
    }
    extra_candidates.iter().find(|path| path.is_file()).cloned()
}

fn uv_candidates() -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    if let Some(home) = home_dir() {
        candidates.push(home.join(".local/bin/uv"));
        candidates.push(home.join(".cargo/bin/uv"));
        candidates.push(home.join(".local/bin/uv.exe"));
        candidates.push(home.join(".cargo/bin/uv.exe"));
    }
    candidates.push(PathBuf::from("/opt/homebrew/bin/uv"));
    candidates.push(PathBuf::from("/usr/local/bin/uv"));
    candidates
}

fn uv_path() -> Option<PathBuf> {
    search_executable(&["uv", "uv.exe"], &uv_candidates())
}

fn python_path() -> Option<PathBuf> {
    search_executable(
        &["python3", "python", "py.exe", "python.exe"],
        &[
            PathBuf::from("/opt/homebrew/bin/python3"),
            PathBuf::from("/usr/local/bin/python3"),
        ],
    )
}

fn ffmpeg_path() -> Option<PathBuf> {
    if let Some(configured) = std::env::var_os("FFMPEG_BINARY") {
        let path = PathBuf::from(configured);
        if path.is_file() {
            return Some(path);
        }
    }
    search_executable(
        &["ffmpeg", "ffmpeg.exe"],
        &[
            project_root().join("models/ffmpeg/ffmpeg"),
            project_root().join("models/ffmpeg/ffmpeg.exe"),
            PathBuf::from("/opt/homebrew/bin/ffmpeg"),
            PathBuf::from("/usr/local/bin/ffmpeg"),
        ],
    )
}

fn run_command(mut command: Command) -> Result<CommandResult, String> {
    let output = command
        .output()
        .map_err(|error| format!("failed to run command: {error}"))?;

    Ok(CommandResult {
        ok: output.status.success(),
        code: output.status.code(),
        stdout: String::from_utf8_lossy(&output.stdout).to_string(),
        stderr: String::from_utf8_lossy(&output.stderr).to_string(),
    })
}

fn command_version(path: Option<PathBuf>, args: &[&str]) -> (bool, String) {
    let Some(path) = path else {
        return (false, "missing".to_string());
    };
    let mut command = Command::new(path);
    command.args(args);
    match run_command(command) {
        Ok(result) => {
            let text = [result.stdout.trim(), result.stderr.trim()]
                .into_iter()
                .filter(|part| !part.is_empty())
                .collect::<Vec<_>>()
                .join("\n");
            (result.ok, if text.is_empty() { "available".to_string() } else { text })
        }
        Err(error) => (false, error),
    }
}

fn ensure_uv() -> Result<PathBuf, String> {
    if let Some(path) = uv_path() {
        return Ok(path);
    }

    let install_result = if cfg!(target_os = "windows") {
        let mut command = Command::new("powershell");
        command.args([
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            "irm https://astral.sh/uv/install.ps1 | iex",
        ]);
        run_command(command)
    } else {
        let mut command = Command::new("sh");
        command.arg("-lc").arg("curl -LsSf https://astral.sh/uv/install.sh | sh");
        run_command(command)
    };

    match install_result {
        Ok(result) if result.ok => uv_path().ok_or_else(|| "uv installer finished but uv was not found on PATH".to_string()),
        Ok(result) => Err([result.stdout.trim(), result.stderr.trim()]
            .into_iter()
            .filter(|part| !part.is_empty())
            .collect::<Vec<_>>()
            .join("\n")),
        Err(error) => Err(error),
    }
}

fn detect_profile(profile: Option<String>) -> String {
    let requested = profile.unwrap_or_else(|| "auto".to_string());
    if requested != "auto" {
        return requested;
    }
    if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        return "macos".to_string();
    }
    if Command::new("nvidia-smi")
        .arg("-L")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
    {
        return "nvidia".to_string();
    }
    if Command::new("rocm-smi")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
        || Command::new("hipcc")
            .arg("--version")
            .output()
            .map(|output| output.status.success())
            .unwrap_or(false)
        || Path::new("/dev/kfd").exists()
    {
        return "amd".to_string();
    }
    "cpu".to_string()
}

fn gpu_backend(profile: &str) -> &str {
    match profile {
        "nvidia" => "cuda",
        "amd" => "rocm",
        "macos" => "macos-metal",
        "cpu" => "cpu",
        _ => "unknown",
    }
}

fn dotenv_value(value: &str) -> String {
    if value
        .chars()
        .all(|character| character.is_ascii_alphanumeric() || matches!(character, '/' | '\\' | ':' | '.' | '_' | '-'))
    {
        return value.to_string();
    }
    format!(
        "\"{}\"",
        value
            .replace('\\', "\\\\")
            .replace('"', "\\\"")
            .replace('\n', "\\n")
    )
}

fn runtime_paths() -> Result<(PathBuf, PathBuf, PathBuf), String> {
    let root = project_root();
    let data_dir = root.join("data");
    let models_dir = root.join("models");
    let logs_dir = data_dir.join("logs");
    fs::create_dir_all(&data_dir).map_err(|error| error.to_string())?;
    fs::create_dir_all(&models_dir).map_err(|error| error.to_string())?;
    fs::create_dir_all(&logs_dir).map_err(|error| error.to_string())?;
    Ok((data_dir, models_dir, logs_dir))
}

fn write_native_env(config: &NativeRuntimeConfig, ffmpeg_bin: &Path) -> Result<PathBuf, String> {
    let root = project_root();
    let (data_dir, models_dir, _) = runtime_paths()?;
    let profile = detect_profile(config.runtime_profile.clone());
    let tier = config.model_tier.clone().unwrap_or_else(|| "default".to_string());
    let reasoning_mode = if tier == "quality" { "low" } else { "off" };
    let clips_dir = config
        .clips_dir
        .clone()
        .unwrap_or_else(|| root.join("clips").to_string_lossy().to_string());
    let content = format!(
        concat!(
            "CLIPS_DIR={clips_dir}\n",
            "HOST_CLIPS_DIR={clips_dir}\n",
            "DATA_DIR={data_dir}\n",
            "MODELS_DIR={models_dir}\n",
            "MODEL_TIER={tier}\n",
            "INDEXING_PROFILE=balanced\n",
            "RUNTIME_PROFILE={profile}\n",
            "GPU_BACKEND={gpu_backend}\n",
            "ASR_LANGUAGE=auto\n",
            "SEARCH_MIN_SCORE=0.35\n",
            "SEARCH_CANDIDATE_LIMIT=100\n",
            "QDRANT_URL=local\n",
            "ALLOW_MOCK_MODELS=false\n",
            "AUTO_DOWNLOAD_MODELS=true\n",
            "FFMPEG_BINARY={ffmpeg_bin}\n",
            "HF_HOME={hf_home}\n",
            "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True\n",
            "QWEN_REASONING_MODE={reasoning_mode}\n",
            "QWEN_REASONING_BUDGET_TOKENS=1024\n",
            "ENABLE_TRANSCRIPTION=true\n",
            "ENABLE_RERANKING=true\n",
            "ENABLE_DEEP_REASONING=false\n",
            "ENABLE_AUDIO_EVENT_DETECTION=false\n",
            "BACKEND_HOST=127.0.0.1\n",
            "BACKEND_PORT={backend_port}\n"
        ),
        clips_dir = dotenv_value(&clips_dir),
        data_dir = dotenv_value(&data_dir.to_string_lossy()),
        models_dir = dotenv_value(&models_dir.to_string_lossy()),
        hf_home = dotenv_value(&models_dir.join("huggingface").to_string_lossy()),
        ffmpeg_bin = dotenv_value(&ffmpeg_bin.to_string_lossy()),
        tier = tier,
        reasoning_mode = reasoning_mode,
        profile = profile,
        gpu_backend = gpu_backend(&profile),
        backend_port = config.backend_port.unwrap_or(8000),
    );
    let env_path = root.join(".env");
    fs::write(&env_path, content).map_err(|error| error.to_string())?;
    Ok(env_path)
}

fn sync_python_environment(uv: &Path, profile: &str) -> Result<CommandResult, String> {
    let root = project_root();
    let mut command = Command::new(uv);
    command
        .current_dir(&root)
        .arg("sync")
        .arg("--project")
        .arg(&root)
        .arg("--no-dev");
    let sync_result = run_command(command)?;
    if !sync_result.ok {
        return Ok(sync_result);
    }

    if profile == "nvidia" && cfg!(all(target_os = "linux", target_arch = "x86_64")) {
        let python = root.join(".venv/bin/python");
        let mut fast_path = Command::new(uv);
        fast_path
            .current_dir(&root)
            .arg("pip")
            .arg("install")
            .arg("--python")
            .arg(&python)
            .arg("causal-conv1d")
            .arg("flash-linear-attention>=0.5.0,<1.0");
        let fast_path_result = run_command(fast_path)?;
        return Ok(CommandResult {
            ok: fast_path_result.ok,
            code: fast_path_result.code,
            stdout: [sync_result.stdout, fast_path_result.stdout].join("\n"),
            stderr: [sync_result.stderr, fast_path_result.stderr].join("\n"),
        });
    }

    Ok(sync_result)
}

fn ensure_ffmpeg_with_uv(uv: &Path, models_dir: &Path) -> Result<PathBuf, String> {
    let root = project_root();
    let ffmpeg_dir = models_dir.join("ffmpeg");
    let mut command = Command::new(uv);
    command
        .current_dir(&root)
        .arg("run")
        .arg("--project")
        .arg(&root)
        .arg("python")
        .arg("-c")
        .arg(
            "import os; from pathlib import Path; \
             from app.runtime_tools import ensure_ffmpeg; \
             print(ensure_ffmpeg(Path(os.environ['INSTANT_REPLAY_FFMPEG_DIR'])))",
        )
        .env("INSTANT_REPLAY_FFMPEG_DIR", &ffmpeg_dir);
    let result = run_command(command)?;
    if !result.ok {
        return Err([result.stdout.trim(), result.stderr.trim()]
            .into_iter()
            .filter(|part| !part.is_empty())
            .collect::<Vec<_>>()
            .join("\n"));
    }
    let path_text = result
        .stdout
        .lines()
        .rev()
        .find(|line| !line.trim().is_empty())
        .ok_or_else(|| "FFmpeg installer completed without returning a binary path".to_string())?
        .trim();
    let path = PathBuf::from(path_text);
    if !path.is_file() {
        return Err(format!("FFmpeg installer returned a missing binary path: {path_text}"));
    }
    Ok(path)
}

fn backend_port(config: &NativeRuntimeConfig) -> u16 {
    config.backend_port.unwrap_or(8000)
}

fn is_backend_port_open(port: u16) -> bool {
    std::net::TcpStream::connect(("127.0.0.1", port)).is_ok()
}

#[tauri::command]
fn select_folder() -> Result<Option<String>, String> {
    Ok(rfd::FileDialog::new()
        .pick_folder()
        .map(|path| path.to_string_lossy().to_string()))
}

#[tauri::command]
fn start_native_runtime(
    state: tauri::State<ProcessState>,
    config: NativeRuntimeConfig,
) -> Result<CommandResult, String> {
    let port = backend_port(&config);
    if state.backend.lock().map_err(|error| error.to_string())?.is_some() || is_backend_port_open(port) {
        return Ok(CommandResult {
            ok: true,
            code: Some(0),
            stdout: format!("Native backend is already running on 127.0.0.1:{port}"),
            stderr: String::new(),
        });
    }

    let uv = ensure_uv()?;
    let profile = detect_profile(config.runtime_profile.clone());
    let sync_result = sync_python_environment(&uv, &profile)?;
    if !sync_result.ok {
        return Ok(sync_result);
    }

    let root = project_root();
    let (data_dir, models_dir, logs_dir) = runtime_paths()?;
    let ffmpeg_bin = ensure_ffmpeg_with_uv(&uv, &models_dir)?;
    let env_path = write_native_env(&config, &ffmpeg_bin)?;
    let log_file = logs_dir.join("backend.log");
    let stdout = File::create(&log_file).map_err(|error| error.to_string())?;
    let stderr = stdout.try_clone().map_err(|error| error.to_string())?;
    let tier = config.model_tier.clone().unwrap_or_else(|| "default".to_string());
    let clips_dir = config
        .clips_dir
        .clone()
        .unwrap_or_else(|| root.join("clips").to_string_lossy().to_string());

    let mut command = Command::new(&uv);
    command
        .current_dir(&root)
        .arg("run")
        .arg("--project")
        .arg(&root)
        .arg("python")
        .arg("-m")
        .arg("app")
        .arg("serve")
        .arg("--host")
        .arg("127.0.0.1")
        .arg("--port")
        .arg(port.to_string())
        .env("CLIPS_DIR", clips_dir)
        .env("HOST_CLIPS_DIR", config.clips_dir.unwrap_or_else(|| root.join("clips").to_string_lossy().to_string()))
        .env("DATA_DIR", data_dir)
        .env("MODELS_DIR", &models_dir)
        .env("MODEL_TIER", tier)
        .env("INDEXING_PROFILE", "balanced")
        .env("RUNTIME_PROFILE", &profile)
        .env("GPU_BACKEND", gpu_backend(&profile))
        .env("QDRANT_URL", "local")
        .env("SEARCH_MIN_SCORE", "0.35")
        .env("SEARCH_CANDIDATE_LIMIT", "100")
        .env("ENABLE_RERANKING", "true")
        .env("ALLOW_MOCK_MODELS", "false")
        .env("AUTO_DOWNLOAD_MODELS", "true")
        .env("FFMPEG_BINARY", &ffmpeg_bin)
        .env("HF_HOME", models_dir.join("huggingface"))
        .env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        .env("PYTHONUNBUFFERED", "1")
        .stdout(Stdio::from(stdout))
        .stderr(Stdio::from(stderr));

    let child = command
        .spawn()
        .map_err(|error| format!("failed to start backend through uv: {error}"))?;
    *state.backend.lock().map_err(|error| error.to_string())? = Some(child);

    Ok(CommandResult {
        ok: true,
        code: Some(0),
        stdout: format!(
            "Wrote {}\nuv sync completed.\nFFmpeg ready: {}\nNative backend started on http://127.0.0.1:{port}\nLogs: {}",
            env_path.to_string_lossy(),
            ffmpeg_bin.to_string_lossy(),
            log_file.to_string_lossy()
        ),
        stderr: String::new(),
    })
}

#[tauri::command]
fn stop_native_runtime(state: tauri::State<ProcessState>) -> Result<CommandResult, String> {
    let mut child = state.backend.lock().map_err(|error| error.to_string())?.take();
    if let Some(process) = child.as_mut() {
        let _ = process.kill();
        let _ = process.wait();
        Ok(CommandResult {
            ok: true,
            code: Some(0),
            stdout: "Native backend stopped.".to_string(),
            stderr: String::new(),
        })
    } else {
        Ok(CommandResult {
            ok: true,
            code: Some(0),
            stdout: "No app-managed native backend process is running.".to_string(),
            stderr: String::new(),
        })
    }
}

#[tauri::command]
fn suggested_folders() -> Vec<String> {
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .unwrap_or_else(|_| ".".to_string());
    if cfg!(target_os = "windows") {
        vec![
            format!(r"{home}\Videos\NVIDIA\Instant Replay"),
            format!(r"{home}\Videos\NVIDIA"),
            format!(r"{home}\Videos\Captures"),
            format!(r"{home}\Videos"),
        ]
    } else if cfg!(target_os = "macos") {
        let test_videos = format!("{home}/nvidia test videos");
        let mut folders = Vec::new();
        if Path::new(&test_videos).exists() {
            folders.push(test_videos);
        }
        folders.extend([
            format!("{home}/Movies"),
            format!("{home}/Movies/NVIDIA"),
            format!("{home}/Videos"),
            "./clips".to_string(),
        ]);
        folders
    } else {
        vec![
            format!("{home}/Videos/NVIDIA"),
            format!("{home}/Videos/Captures"),
            format!("{home}/Videos"),
            "./clips".to_string(),
        ]
    }
}

#[tauri::command]
fn probe_runtime() -> RuntimeProbe {
    let root = project_root();
    let (data_dir, models_dir, logs_dir) = runtime_paths().unwrap_or_else(|_| {
        (
            root.join("data"),
            root.join("models"),
            root.join("data").join("logs"),
        )
    });
    let (uv_available, uv_version) = command_version(uv_path(), &["--version"]);
    let (python_available, python_version) = command_version(python_path(), &["--version"]);
    let (ffmpeg_available, ffmpeg_version) = command_version(ffmpeg_path(), &["-version"]);
    RuntimeProbe {
        uv_available,
        uv_version,
        python_available,
        python_version,
        ffmpeg_available,
        ffmpeg_version: ffmpeg_version.lines().next().unwrap_or("missing").to_string(),
        backend_running: is_backend_port_open(8000),
        project_root: root.to_string_lossy().to_string(),
        data_dir: data_dir.to_string_lossy().to_string(),
        models_dir: models_dir.to_string_lossy().to_string(),
        log_file: logs_dir.join("backend.log").to_string_lossy().to_string(),
    }
}

pub fn run() {
    tauri::Builder::default()
        .manage(ProcessState::default())
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![
            select_folder,
            start_native_runtime,
            stop_native_runtime,
            suggested_folders,
            probe_runtime
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
