import {
  AlertTriangle,
  Check,
  ChevronRight,
  CircleStop,
  Clock3,
  Cpu,
  Database,
  Folder,
  Gauge,
  HardDrive,
  ListRestart,
  Loader2,
  Play,
  RefreshCw,
  Save,
  Search,
  Server,
  Settings,
  SquareTerminal,
  Tags,
  Video,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createApi } from "./api/client";
import { Sidebar } from "./components/Sidebar";
import { StatusPill } from "./components/StatusPill";
import { usePersistentState } from "./state/usePersistentState";
import { nativeRuntimeLabel, probeRuntime, selectFolder, startNativeRuntime, stopNativeRuntime, suggestedFolders } from "./services/tauriCommands";
import type {
  AppSettings,
  BackendHealth,
  ClipDetail,
  ClipGroup,
  DashboardStats,
  IndexJob,
  LogEntry,
  ModelTier,
  RuntimeProbe,
  RuntimeProfile,
  SearchResult,
  ViewId,
} from "./types";

const defaultSettings: AppSettings = {
  backendUrl: "http://127.0.0.1:8000",
  captureFolders: [],
  exportFolder: "",
  runtimeProfile: "auto",
  modelTier: "default",
  setupComplete: false,
  autoStartBackend: true,
  hardwareAcceleration: true,
};

const fallbackStats: DashboardStats = {
  clipCount: 0,
  indexedClipCount: 0,
  groupCount: 0,
  queuedJobs: 0,
  activeJobs: 0,
  storageGb: 0,
};

const fallbackGroups: ClipGroup[] = [
  {
    id: "clutch",
    name: "Clutch plays",
    color: "#3b82f6",
    clipCount: 0,
    lastUpdated: "Waiting for index",
    rules: ["overtime", "low health", "last teamfight"],
  },
  {
    id: "mechanics",
    name: "Mechanical mistakes",
    color: "#f97316",
    clipCount: 0,
    lastUpdated: "Waiting for index",
    rules: ["missed ability", "bad reload", "poor crosshair placement"],
  },
  {
    id: "teamplay",
    name: "Team coordination",
    color: "#10b981",
    clipCount: 0,
    lastUpdated: "Waiting for index",
    rules: ["rotation", "utility combo", "voice callout"],
  },
];

const fallbackJobs: IndexJob[] = [];

const fallbackLogs: LogEntry[] = [
  {
    id: "boot",
    level: "info",
    source: "native-ui",
    message: "UI initialized. Backend logs will appear when localhost API is reachable.",
    time: new Date().toLocaleTimeString(),
  },
];

const sampleResults: SearchResult[] = [
  {
    id: "preview-ace",
    title: "Preview result: late round retake",
    path: "clips/preview/retake.mp4",
    game: "Unindexed",
    timestamp: "00:42",
    durationSec: 38,
    score: 0.82,
    tags: ["retake", "utility", "crossfire"],
    summary: "Local preview item shown until the backend returns indexed clips.",
  },
];

function runtimeProfileLabel(profile: RuntimeProfile) {
  return {
    auto: "Auto",
    nvidia: "NVIDIA",
    amd: "AMD",
    macos: "macOS / Apple Silicon",
    cpu: "CPU",
  }[profile];
}

function tierLabel(tier: ModelTier) {
  return {
    default: "Default",
    quality: "Quality",
  }[tier];
}

function progressPercent(value: number) {
  return `${Math.max(0, Math.min(100, Math.round(value * 100)))}%`;
}

function sleep(ms: number) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

function runtimeStartKey(settings: AppSettings) {
  return [
    settings.captureFolders[0] ?? "",
    settings.runtimeProfile,
    settings.modelTier,
    settings.backendUrl,
  ].join("|");
}

export function App() {
  const [settings, setSettings] = usePersistentState<AppSettings>("ira.native-ui.settings", defaultSettings);
  const [activeView, setActiveView] = useState<ViewId>(settings.setupComplete ? "dashboard" : "setup");
  const [health, setHealth] = useState<BackendHealth>({
    state: "checking",
    message: "Checking backend",
  });
  const [stats, setStats] = useState<DashboardStats>(fallbackStats);
  const [groups, setGroups] = useState<ClipGroup[]>(fallbackGroups);
  const [jobs, setJobs] = useState<IndexJob[]>(fallbackJobs);
  const [logs, setLogs] = useState<LogEntry[]>(fallbackLogs);
  const [results, setResults] = useState<SearchResult[]>(sampleResults);
  const [selectedClip, setSelectedClip] = useState<ClipDetail | null>(null);
  const [searchQuery, setSearchQuery] = useState("triple kill after rotation");
  const [selectedGroup, setSelectedGroup] = useState("all");
  const [runtimeProbe, setRuntimeProbe] = useState<RuntimeProbe | null>(null);
  const [commandOutput, setCommandOutput] = useState("");
  const [busyAction, setBusyAction] = useState<string | null>(null);
  const autoStartKeyRef = useRef<string | null>(null);

  const api = useMemo(() => createApi({ backendUrl: settings.backendUrl }), [settings.backendUrl]);

  const refreshBackend = useCallback(async () => {
    const nextHealth = await api.health();
    setHealth(nextHealth);

    if (nextHealth.state === "offline") {
      return;
    }

    const [nextStats, nextGroups, nextJobs, nextLogs] = await Promise.allSettled([
      api.dashboard(),
      api.groups(),
      api.indexJobs(),
      api.logs(),
    ]);

    if (nextStats.status === "fulfilled") setStats(nextStats.value);
    if (nextGroups.status === "fulfilled") setGroups(nextGroups.value);
    if (nextJobs.status === "fulfilled") setJobs(nextJobs.value);
    if (nextLogs.status === "fulfilled") setLogs(nextLogs.value);
  }, [api]);

  useEffect(() => {
    void refreshBackend();
    const timer = window.setInterval(refreshBackend, 15_000);
    return () => window.clearInterval(timer);
  }, [refreshBackend]);

  const waitForBackend = useCallback(
    async (timeoutMs = 180_000) => {
      const deadline = Date.now() + timeoutMs;
      let lastHealth: BackendHealth = {
        state: "checking",
        message: "Waiting for backend API",
      };
      while (Date.now() < deadline) {
        lastHealth = await api.health();
        setHealth(lastHealth);
        if (lastHealth.state !== "offline") {
          return lastHealth;
        }
        await sleep(2_000);
      }
      return lastHealth;
    },
    [api],
  );

  const startRuntime = useCallback(
    async (targetSettings: AppSettings, busyLabel: string = "start") => {
      const folder = targetSettings.captureFolders[0];
      if (!folder) {
        setCommandOutput("Select a clips folder before starting the native backend.");
        return false;
      }

      const key = runtimeStartKey(targetSettings);
      autoStartKeyRef.current = key;
      setBusyAction(busyLabel);
      setCommandOutput("Preparing local runtime...");

      try {
        const probe = await probeRuntime();
        setRuntimeProbe(probe);
        if (!probe.python_available && !probe.uv_available) {
          throw new Error(`Python and uv are unavailable. Python: ${probe.python_version}; uv: ${probe.uv_version}`);
        }

        setCommandOutput("Installing native dependencies with uv and starting backend...");
        const result = await startNativeRuntime({
          clipsDir: folder,
          runtimeProfile: targetSettings.runtimeProfile,
          modelTier: targetSettings.modelTier,
          backendPort: 8000,
        });
        const output = [result.stdout.trim(), result.stderr.trim()].filter(Boolean).join("\n");
        if (!result.ok) {
          throw new Error(output || `native runtime exited with code ${result.code ?? "unknown"}`);
        }

        setCommandOutput([output || "Native backend started.", "Waiting for backend health check..."].join("\n"));
        const nextHealth = await waitForBackend();
        await refreshBackend();
        setCommandOutput(
          [
            output || "Native backend started.",
            nextHealth.state === "offline" ? `Backend did not become ready: ${nextHealth.message}` : "Backend API is ready.",
          ].join("\n"),
        );
        return nextHealth.state !== "offline";
      } catch (error) {
        setCommandOutput(error instanceof Error ? error.message : "Unable to start local runtime.");
        return false;
      } finally {
        setBusyAction((current) => (current === busyLabel ? null : current));
      }
    },
    [refreshBackend, waitForBackend],
  );

  useEffect(() => {
    probeRuntime()
      .then(setRuntimeProbe)
      .catch((error) =>
        setRuntimeProbe({
          uv_available: false,
          uv_version: error instanceof Error ? error.message : "Native probe unavailable",
          python_available: false,
          python_version: "Not checked",
          ffmpeg_available: false,
          ffmpeg_version: "Not checked",
          backend_running: false,
          project_root: "",
          data_dir: "",
          models_dir: "",
          log_file: "",
        }),
      );
  }, []);

  useEffect(() => {
    if (settings.captureFolders.length > 0) return;
    suggestedFolders()
      .then((folders) => {
        if (folders.length) patchSettings({ captureFolders: [folders[0]] });
      })
      .catch(() => undefined);
  }, []);

  useEffect(() => {
    if (!settings.autoStartBackend || !settings.captureFolders[0]) return;
    const key = runtimeStartKey(settings);
    if (autoStartKeyRef.current === key) return;
    autoStartKeyRef.current = key;
    void (async () => {
      const currentHealth = await api.health();
      setHealth(currentHealth);
      if (currentHealth.state !== "offline") return;
      await startRuntime(settings, "autostart");
    })();
  }, [api, settings, startRuntime]);

  const patchSettings = (patch: Partial<AppSettings>) => {
    setSettings({ ...settings, ...patch });
  };

  const addCaptureFolder = async () => {
    setBusyAction("folder");
    try {
      const folder = await selectFolder();
      if (folder && !settings.captureFolders.includes(folder)) {
        patchSettings({ captureFolders: [...settings.captureFolders, folder] });
      }
    } catch (error) {
      setCommandOutput(error instanceof Error ? error.message : "Folder picker failed");
    } finally {
      setBusyAction(null);
    }
  };

  const pickExportFolder = async () => {
    setBusyAction("export");
    try {
      const folder = await selectFolder();
      if (folder) patchSettings({ exportFolder: folder });
    } catch (error) {
      setCommandOutput(error instanceof Error ? error.message : "Folder picker failed");
    } finally {
      setBusyAction(null);
    }
  };

  const runRuntime = async (action: "start" | "stop") => {
    if (action === "start") {
      await startRuntime(settings, "start");
      return;
    }
    setBusyAction(action);
    try {
      const result = await stopNativeRuntime();
      setCommandOutput([result.stdout.trim(), result.stderr.trim()].filter(Boolean).join("\n") || `native runtime ${action} completed`);
      await refreshBackend();
    } catch (error) {
      setCommandOutput(error instanceof Error ? error.message : `native runtime ${action} failed`);
    } finally {
      setBusyAction(null);
    }
  };

  const startIndex = async (folder: string) => {
    setBusyAction(`index-${folder}`);
    try {
      const job = await api.startIndexing(folder, settings.runtimeProfile, settings.modelTier);
      setJobs([job, ...jobs.filter((item) => item.id !== job.id)]);
      setActiveView("indexing");
    } catch (error) {
      setCommandOutput(error instanceof Error ? error.message : "Unable to start indexing");
      setActiveView("logs");
    } finally {
      setBusyAction(null);
    }
  };

  const runSearch = async () => {
    setBusyAction("search");
    try {
      const nextResults = await api.search({
        query: searchQuery,
        groupId: selectedGroup === "all" ? undefined : selectedGroup,
        minScore: 0.35,
      });
      setResults(nextResults.length ? nextResults : []);
    } catch {
      setResults(searchQuery.trim() ? sampleResults : []);
    } finally {
      setBusyAction(null);
    }
  };

  const openClip = async (clip: SearchResult) => {
    setActiveView("clip");
    try {
      const detail = await api.clip(clip.id);
      setSelectedClip(detail);
    } catch {
      setSelectedClip({
        ...clip,
        transcript: ["Preview transcript placeholder. Run indexing to populate semantic events and speech cues."],
        events: [
          { time: "00:08", label: "Rotation detected", confidence: 0.71 },
          { time: "00:19", label: "Engagement spike", confidence: 0.82 },
          { time: "00:31", label: "Round outcome", confidence: 0.77 },
        ],
        technical: {
          resolution: "1920x1080",
          fps: 60,
          codec: "h264",
          sizeMb: 48,
        },
      });
    }
  };

  const completeSetup = async () => {
    const next = { ...settings, setupComplete: true };
    setSettings(next);
    setActiveView("dashboard");
    api.saveSettings(next).catch(() => undefined);
    if (next.autoStartBackend) {
      void startRuntime(next, "start");
    }
  };

  const backendLabel = `${health.state.toUpperCase()} ${health.latencyMs ? `${health.latencyMs}ms` : ""}`.trim();

  return (
    <div className="app-shell">
      <Sidebar
        activeView={activeView}
        onNavigate={setActiveView}
        backendLabel={backendLabel}
        setupComplete={settings.setupComplete}
      />

      <main className="workspace">
        <header className="topbar">
          <div>
            <div className="eyebrow">Local runtime</div>
            <h1>{viewTitle(activeView)}</h1>
          </div>
          <div className="topbar-actions">
            <StatusPill state={health.state} label={backendLabel} />
            <button className="icon-button" type="button" onClick={refreshBackend} title="Refresh status">
              <RefreshCw size={16} />
            </button>
          </div>
        </header>

        {activeView === "setup" && (
          <SetupView
            settings={settings}
            busyAction={busyAction}
            onPatch={patchSettings}
            onPickCapture={addCaptureFolder}
            onPickExport={pickExportFolder}
            onComplete={completeSetup}
          />
        )}

        {activeView === "dashboard" && (
          <DashboardView
            stats={stats}
            health={health}
            jobs={jobs}
            groups={groups}
            settings={settings}
            onNavigate={setActiveView}
          />
        )}

        {activeView === "groups" && <GroupsView groups={groups} />}

        {activeView === "indexing" && (
          <IndexingView
            settings={settings}
            jobs={jobs}
            busyAction={busyAction}
            onPickCapture={addCaptureFolder}
            onStartIndex={startIndex}
            onRemoveFolder={(folder) =>
              patchSettings({ captureFolders: settings.captureFolders.filter((item) => item !== folder) })
            }
          />
        )}

        {activeView === "search" && (
          <SearchView
            query={searchQuery}
            selectedGroup={selectedGroup}
            groups={groups}
            results={results}
            busyAction={busyAction}
            onQueryChange={setSearchQuery}
            onGroupChange={setSelectedGroup}
            onSearch={runSearch}
            onOpenClip={openClip}
          />
        )}

        {activeView === "clip" && <ClipDetailView clip={selectedClip} />}

        {activeView === "settings" && (
          <SettingsView settings={settings} onPatch={patchSettings} onPickExport={pickExportFolder} busyAction={busyAction} />
        )}

        {activeView === "logs" && (
          <LogsView
            health={health}
            logs={logs}
            runtimeProbe={runtimeProbe}
            commandOutput={commandOutput}
            settings={settings}
            busyAction={busyAction}
            onPatch={patchSettings}
            onStart={() => runRuntime("start")}
            onStop={() => runRuntime("stop")}
            onProbe={() => probeRuntime().then(setRuntimeProbe).catch((error) => setCommandOutput(String(error)))}
          />
        )}
      </main>
    </div>
  );
}

function viewTitle(view: ViewId) {
  return {
    setup: "First-run setup",
    dashboard: "Dashboard",
    groups: "Clip groups",
    indexing: "Indexing",
    search: "Semantic search",
    clip: "Clip detail",
    settings: "Settings",
    logs: "Logs and runtime status",
  }[view];
}

function SetupView({
  settings,
  busyAction,
  onPatch,
  onPickCapture,
  onPickExport,
  onComplete,
}: {
  settings: AppSettings;
  busyAction: string | null;
  onPatch: (patch: Partial<AppSettings>) => void;
  onPickCapture: () => void;
  onPickExport: () => void;
  onComplete: () => void;
}) {
  return (
    <section className="content-grid content-grid-setup">
      <div className="panel panel-wide">
        <div className="panel-header">
          <div>
            <h2>Setup checklist</h2>
            <p>Connect folders, select a runtime profile, and bind the UI to the local API.</p>
          </div>
          <StatusPill state={settings.setupComplete ? "complete" : "queued"} label={settings.setupComplete ? "Complete" : "Pending"} />
        </div>

        <div className="setup-steps">
          <SetupStep done={settings.captureFolders.length > 0} label="Capture folders" value={`${settings.captureFolders.length} configured`} />
          <SetupStep done={Boolean(settings.exportFolder)} label="Export folder" value={settings.exportFolder || "Not selected"} />
          <SetupStep done={Boolean(settings.backendUrl)} label="Backend API" value={settings.backendUrl} />
          <SetupStep done label="Runtime" value={`${runtimeProfileLabel(settings.runtimeProfile)} / ${tierLabel(settings.modelTier)}`} />
        </div>

        <div className="form-section">
          <label>
            Backend URL
            <input value={settings.backendUrl} onChange={(event) => onPatch({ backendUrl: event.target.value })} />
          </label>
        </div>

        <div className="button-row">
          <button className="primary-button" type="button" onClick={onPickCapture} disabled={busyAction === "folder"}>
            <Folder size={16} />
            Add capture folder
          </button>
          <button className="secondary-button" type="button" onClick={onPickExport} disabled={busyAction === "export"}>
            <HardDrive size={16} />
            Pick export folder
          </button>
          <button className="primary-button accent" type="button" onClick={onComplete}>
            <Check size={16} />
            Finish setup
          </button>
        </div>
      </div>

      <RuntimeSelector settings={settings} onPatch={onPatch} />
    </section>
  );
}

function SetupStep({ done, label, value }: { done: boolean; label: string; value: string }) {
  return (
    <div className="setup-step">
      <span className={done ? "step-dot step-dot-done" : "step-dot"}>{done ? <Check size={13} /> : <Clock3 size={13} />}</span>
      <div>
        <strong>{label}</strong>
        <span>{value}</span>
      </div>
    </div>
  );
}

function RuntimeSelector({
  settings,
  onPatch,
}: {
  settings: AppSettings;
  onPatch: (patch: Partial<AppSettings>) => void;
}) {
  const profiles: RuntimeProfile[] = ["auto", "nvidia", "amd", "macos", "cpu"];
  const tiers: ModelTier[] = ["default", "quality"];

  return (
    <div className="panel">
      <div className="panel-header compact">
        <h2>Runtime and models</h2>
        <Cpu size={18} />
      </div>
      <div className="control-group">
        <span className="control-label">Runtime profile</span>
        <div className="segmented">
          {profiles.map((profile) => (
            <button
              key={profile}
              type="button"
              className={settings.runtimeProfile === profile ? "selected" : ""}
              onClick={() => onPatch({ runtimeProfile: profile })}
            >
              {runtimeProfileLabel(profile)}
            </button>
          ))}
        </div>
      </div>
      <div className="control-group">
        <span className="control-label">Model tier</span>
        <div className="option-list">
          {tiers.map((tier) => (
            <button
              key={tier}
              type="button"
              className={settings.modelTier === tier ? "option-row selected" : "option-row"}
              onClick={() => onPatch({ modelTier: tier })}
            >
              <span>{tierLabel(tier)}</span>
              <ChevronRight size={15} />
            </button>
          ))}
        </div>
      </div>
      <label className="toggle-row">
        <input
          type="checkbox"
          checked={settings.hardwareAcceleration}
          onChange={(event) => onPatch({ hardwareAcceleration: event.target.checked })}
        />
        <span>Use hardware acceleration when available</span>
      </label>
    </div>
  );
}

function DashboardView({
  stats,
  health,
  jobs,
  groups,
  settings,
  onNavigate,
}: {
  stats: DashboardStats;
  health: BackendHealth;
  jobs: IndexJob[];
  groups: ClipGroup[];
  settings: AppSettings;
  onNavigate: (view: ViewId) => void;
}) {
  return (
    <section className="dashboard-layout">
      <div className="metric-grid">
        <Metric icon={Video} label="Clips" value={stats.clipCount.toLocaleString()} sub={`${stats.indexedClipCount} indexed`} />
        <Metric icon={Tags} label="Groups" value={stats.groupCount.toLocaleString()} sub={`${groups.length} visible`} />
        <Metric icon={Database} label="Queue" value={`${stats.activeJobs}/${stats.queuedJobs}`} sub="active / queued" />
        <Metric icon={HardDrive} label="Storage" value={`${stats.storageGb.toFixed(1)} GB`} sub="local metadata" />
      </div>

      <div className="content-grid">
        <div className="panel panel-wide">
          <div className="panel-header">
            <div>
              <h2>Backend</h2>
              <p>{settings.backendUrl}</p>
            </div>
            <StatusPill state={health.state} label={health.state} />
          </div>
          <div className="status-strip">
            <span>{health.message}</span>
            <span>{nativeRuntimeLabel()}</span>
            <span>{runtimeProfileLabel(settings.runtimeProfile)}</span>
            <span>{tierLabel(settings.modelTier)}</span>
          </div>
          <div className="button-row">
            <button className="secondary-button" type="button" onClick={() => onNavigate("indexing")}>
              <Database size={16} />
              Open indexing
            </button>
            <button className="secondary-button" type="button" onClick={() => onNavigate("search")}>
              <Search size={16} />
              Search clips
            </button>
            <button className="secondary-button" type="button" onClick={() => onNavigate("logs")}>
              <SquareTerminal size={16} />
              Runtime status
            </button>
          </div>
        </div>

        <div className="panel">
          <div className="panel-header compact">
            <h2>Active jobs</h2>
            <Loader2 size={18} />
          </div>
          <JobList jobs={jobs.slice(0, 4)} />
        </div>
      </div>
    </section>
  );
}

function Metric({ icon: Icon, label, value, sub }: { icon: typeof Video; label: string; value: string; sub: string }) {
  return (
    <div className="metric-tile">
      <Icon size={18} />
      <div>
        <span>{label}</span>
        <strong>{value}</strong>
        <small>{sub}</small>
      </div>
    </div>
  );
}

function GroupsView({ groups }: { groups: ClipGroup[] }) {
  return (
    <section className="table-panel">
      <div className="panel-header">
        <div>
          <h2>Groups</h2>
          <p>Rule-backed semantic buckets for review workflows.</p>
        </div>
        <button className="secondary-button" type="button">
          <ListRestart size={16} />
          Refresh
        </button>
      </div>
      <div className="data-table">
        <div className="table-row table-head">
          <span>Name</span>
          <span>Rules</span>
          <span>Clips</span>
          <span>Updated</span>
        </div>
        {groups.map((group) => (
          <div className="table-row" key={group.id}>
            <span className="group-name">
              <i style={{ background: group.color }} />
              {group.name}
            </span>
            <span>{group.rules.join(", ")}</span>
            <span>{group.clipCount}</span>
            <span>{group.lastUpdated}</span>
          </div>
        ))}
      </div>
    </section>
  );
}

function IndexingView({
  settings,
  jobs,
  busyAction,
  onPickCapture,
  onStartIndex,
  onRemoveFolder,
}: {
  settings: AppSettings;
  jobs: IndexJob[];
  busyAction: string | null;
  onPickCapture: () => void;
  onStartIndex: (folder: string) => void;
  onRemoveFolder: (folder: string) => void;
}) {
  return (
    <section className="content-grid">
      <div className="panel panel-wide">
        <div className="panel-header">
          <div>
            <h2>Capture folders</h2>
            <p>Folders are indexed locally and sent to the semantic backend on localhost.</p>
          </div>
          <button className="primary-button" type="button" onClick={onPickCapture} disabled={busyAction === "folder"}>
            <Folder size={16} />
            Add folder
          </button>
        </div>
        <div className="folder-list">
          {settings.captureFolders.length === 0 && <div className="empty-state">No capture folders selected.</div>}
          {settings.captureFolders.map((folder) => (
            <div className="folder-row" key={folder}>
              <Folder size={16} />
              <span>{folder}</span>
              <button className="secondary-button" type="button" onClick={() => onStartIndex(folder)} disabled={busyAction === `index-${folder}`}>
                <Play size={15} />
                Index
              </button>
              <button className="icon-button" type="button" title="Remove folder" onClick={() => onRemoveFolder(folder)}>
                <CircleStop size={15} />
              </button>
            </div>
          ))}
        </div>
      </div>
      <div className="panel">
        <div className="panel-header compact">
          <h2>Queue</h2>
          <Gauge size={18} />
        </div>
        <JobList jobs={jobs} />
      </div>
    </section>
  );
}

function JobList({ jobs }: { jobs: IndexJob[] }) {
  if (!jobs.length) return <div className="empty-state">No indexing jobs.</div>;
  return (
    <div className="job-list">
      {jobs.map((job) => (
        <div className="job-row" key={job.id}>
          <div className="job-topline">
            <span>{job.folder}</span>
            <StatusPill state={job.status} label={job.status} />
          </div>
          <div className="progress-track">
            <i style={{ width: progressPercent(job.progress) }} />
          </div>
          <div className="job-meta">
            <span>{job.clipsFound} clips</span>
            <span>{job.startedAt}</span>
          </div>
        </div>
      ))}
    </div>
  );
}

function SearchView({
  query,
  selectedGroup,
  groups,
  results,
  busyAction,
  onQueryChange,
  onGroupChange,
  onSearch,
  onOpenClip,
}: {
  query: string;
  selectedGroup: string;
  groups: ClipGroup[];
  results: SearchResult[];
  busyAction: string | null;
  onQueryChange: (query: string) => void;
  onGroupChange: (group: string) => void;
  onSearch: () => void;
  onOpenClip: (clip: SearchResult) => void;
}) {
  return (
    <section className="search-layout">
      <div className="search-bar">
        <Search size={18} />
        <input
          value={query}
          onChange={(event) => onQueryChange(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter") void onSearch();
          }}
        />
        <select value={selectedGroup} onChange={(event) => onGroupChange(event.target.value)}>
          <option value="all">All groups</option>
          {groups.map((group) => (
            <option key={group.id} value={group.id}>
              {group.name}
            </option>
          ))}
        </select>
        <button className="primary-button" type="button" onClick={onSearch} disabled={busyAction === "search"}>
          <Search size={16} />
          Run search
        </button>
      </div>

      <div className="results-list">
        {results.map((result) => (
          <button className="result-row" type="button" key={result.id} onClick={() => onOpenClip(result)}>
            <div className="thumbnail-fallback">
              <Video size={20} />
            </div>
            <div className="result-body">
              <div className="result-title">
                <strong>{result.title}</strong>
                <span>{Math.round(result.score * 100)}%</span>
              </div>
              <p>{result.summary}</p>
              <div className="tag-row">
                <span>{result.game}</span>
                {result.timestamp && <span>{result.timestamp}</span>}
                {result.segmentRange ? (
                  <span>{result.segmentRange}</span>
                ) : (
                  result.durationSec > 0 && <span>{result.durationSec}s</span>
                )}
                {result.matchedModality && <span>{result.matchedModality}</span>}
                {result.activeWeapon && <span>{result.activeWeapon}</span>}
                {result.killedByWeapon && <span>Killed with {result.killedByWeapon}</span>}
                {result.tags.map((tag) => (
                  <i key={tag}>{tag}</i>
                ))}
              </div>
              {result.matchedReason && <small className="result-reason">{result.matchedReason}</small>}
            </div>
          </button>
        ))}
        {!results.length && <div className="empty-state">No clips matched the current query.</div>}
      </div>
    </section>
  );
}

function ClipDetailView({ clip }: { clip: ClipDetail | null }) {
  if (!clip) {
    return (
      <section className="panel">
        <div className="empty-state">Select a search result to inspect semantic events, transcript cues, and file details.</div>
      </section>
    );
  }

  return (
    <section className="clip-layout">
      <div className="panel panel-wide">
        <div className="clip-preview">
          <Video size={40} />
          <span>{clip.path}</span>
        </div>
        <div className="panel-header">
          <div>
            <h2>{clip.title}</h2>
            <p>{clip.summary}</p>
          </div>
          <StatusPill state="complete" label={`${Math.round(clip.score * 100)}% match`} />
        </div>
        <div className="tag-row roomy">
          {clip.activeWeapon && <i>{clip.activeWeapon}</i>}
          {clip.killedByWeapon && <i>Killed with {clip.killedByWeapon}</i>}
          {clip.killerName && <i>Killer: {clip.killerName}</i>}
          {(clip.detectedLoadout ?? []).map((item) => (
            <i key={item}>{item}</i>
          ))}
          {clip.tags.map((tag) => (
            <i key={tag}>{tag}</i>
          ))}
        </div>
      </div>
      <div className="panel">
        <div className="panel-header compact">
          <h2>Events</h2>
          <Clock3 size={18} />
        </div>
        <div className="event-list">
          {clip.events.map((event) => (
            <div className="event-row" key={`${event.time}-${event.label}`}>
              <span>{event.time}</span>
              <strong>{event.label}</strong>
              <small>{Math.round(event.confidence * 100)}%</small>
            </div>
          ))}
        </div>
      </div>
      <div className="panel panel-wide">
        <div className="panel-header compact">
          <h2>Transcript cues</h2>
          <Tags size={18} />
        </div>
        {clip.transcript.map((line, index) => (
          <p className="transcript-line" key={`${line}-${index}`}>
            {line}
          </p>
        ))}
      </div>
      <div className="panel">
        <div className="panel-header compact">
          <h2>File</h2>
          <HardDrive size={18} />
        </div>
        <dl className="details-list">
          <dt>Game</dt>
          <dd>{clip.game}</dd>
          <dt>Resolution</dt>
          <dd>{clip.technical.resolution}</dd>
          <dt>FPS</dt>
          <dd>{clip.technical.fps}</dd>
          <dt>Codec</dt>
          <dd>{clip.technical.codec}</dd>
          <dt>Size</dt>
          <dd>{clip.technical.sizeMb} MB</dd>
        </dl>
      </div>
    </section>
  );
}

function SettingsView({
  settings,
  onPatch,
  onPickExport,
  busyAction,
}: {
  settings: AppSettings;
  onPatch: (patch: Partial<AppSettings>) => void;
  onPickExport: () => void;
  busyAction: string | null;
}) {
  return (
    <section className="content-grid">
      <div className="panel panel-wide">
        <div className="panel-header compact">
          <h2>Local API</h2>
          <Server size={18} />
        </div>
        <div className="form-section">
          <label>
            Backend URL
            <input value={settings.backendUrl} onChange={(event) => onPatch({ backendUrl: event.target.value })} />
          </label>
          <label>
            Export folder
            <div className="input-with-button">
              <input value={settings.exportFolder} onChange={(event) => onPatch({ exportFolder: event.target.value })} />
              <button className="icon-button" type="button" onClick={onPickExport} disabled={busyAction === "export"} title="Pick export folder">
                <Folder size={15} />
              </button>
            </div>
          </label>
        </div>
      </div>
      <RuntimeSelector settings={settings} onPatch={onPatch} />
      <div className="panel">
        <div className="panel-header compact">
          <h2>Behavior</h2>
          <Settings size={18} />
        </div>
        <label className="toggle-row">
          <input
            type="checkbox"
            checked={settings.autoStartBackend}
            onChange={(event) => onPatch({ autoStartBackend: event.target.checked })}
          />
          <span>Auto-start native backend</span>
        </label>
        <label className="toggle-row">
          <input
            type="checkbox"
            checked={settings.setupComplete}
            onChange={(event) => onPatch({ setupComplete: event.target.checked })}
          />
          <span>Mark first-run setup complete</span>
        </label>
        <button className="primary-button" type="button">
          <Save size={16} />
          Save local settings
        </button>
      </div>
    </section>
  );
}

function LogsView({
  health,
  logs,
  runtimeProbe,
  commandOutput,
  settings,
  busyAction,
  onPatch,
  onStart,
  onStop,
  onProbe,
}: {
  health: BackendHealth;
  logs: LogEntry[];
  runtimeProbe: RuntimeProbe | null;
  commandOutput: string;
  settings: AppSettings;
  busyAction: string | null;
  onPatch: (patch: Partial<AppSettings>) => void;
  onStart: () => void;
  onStop: () => void;
  onProbe: () => void;
}) {
  return (
    <section className="content-grid">
      <div className="panel">
        <div className="panel-header compact">
          <h2>Runtime</h2>
          <Server size={18} />
        </div>
        <div className="runtime-grid">
          <span>Backend</span>
          <StatusPill state={health.state} label={health.state} />
          <span>uv</span>
          <StatusPill state={runtimeProbe?.uv_available ? "complete" : "failed"} label={runtimeProbe?.uv_available ? "available" : "bootstrap on start"} />
          <span>Python</span>
          <StatusPill state={runtimeProbe?.python_available ? "complete" : "failed"} label={runtimeProbe?.python_available ? "available" : "missing"} />
          <span>FFmpeg</span>
          <StatusPill state={runtimeProbe?.ffmpeg_available ? "complete" : "queued"} label={runtimeProbe?.ffmpeg_available ? "system" : "bundled fallback"} />
        </div>
        <div className="button-row">
          <button className="primary-button" type="button" onClick={onStart} disabled={busyAction === "start"}>
            <Play size={16} />
            Start
          </button>
          <button className="secondary-button" type="button" onClick={onStop} disabled={busyAction === "stop"}>
            <CircleStop size={16} />
            Stop
          </button>
          <button className="secondary-button" type="button" onClick={onProbe}>
            <RefreshCw size={16} />
            Probe
          </button>
        </div>
      </div>
      <div className="panel panel-wide">
        <div className="panel-header compact">
          <h2>Status output</h2>
          <SquareTerminal size={18} />
        </div>
        <pre className="terminal-output">
{[
  `native runtime: ${nativeRuntimeLabel()}`,
  `backend: ${health.message}`,
  `uv: ${runtimeProbe?.uv_version ?? "not probed"}`,
  `python: ${runtimeProbe?.python_version ?? "not probed"}`,
  `ffmpeg: ${runtimeProbe?.ffmpeg_version ?? "not probed"}`,
  `backend process: ${runtimeProbe?.backend_running ? "running" : "not app-managed or stopped"}`,
  `project root: ${runtimeProbe?.project_root ?? ""}`,
  `data: ${runtimeProbe?.data_dir ?? ""}`,
  `models: ${runtimeProbe?.models_dir ?? ""}`,
  `backend log: ${runtimeProbe?.log_file ?? ""}`,
  commandOutput,
]
  .filter(Boolean)
  .join("\n")}
        </pre>
      </div>
      <div className="panel panel-full">
        <div className="panel-header compact">
          <h2>Logs</h2>
          <AlertTriangle size={18} />
        </div>
        <div className="log-list">
          {logs.map((log) => (
            <div className={`log-row log-${log.level}`} key={log.id}>
              <span>{log.time}</span>
              <strong>{log.source}</strong>
              <i>{log.level}</i>
              <p>{log.message}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
