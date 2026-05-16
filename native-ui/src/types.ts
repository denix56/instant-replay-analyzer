export type RuntimeProfile = "auto" | "nvidia" | "amd" | "macos" | "cpu";
export type ModelTier = "default" | "quality";
export type BackendState = "online" | "offline" | "degraded" | "checking";
export type ViewId =
  | "setup"
  | "dashboard"
  | "groups"
  | "indexing"
  | "search"
  | "clip"
  | "settings"
  | "logs";

export interface AppSettings {
  backendUrl: string;
  captureFolders: string[];
  exportFolder: string;
  runtimeProfile: RuntimeProfile;
  modelTier: ModelTier;
  setupComplete: boolean;
  autoStartBackend: boolean;
  hardwareAcceleration: boolean;
}

export interface BackendHealth {
  state: BackendState;
  version?: string;
  message: string;
  latencyMs?: number;
}

export interface DashboardStats {
  clipCount: number;
  indexedClipCount: number;
  groupCount: number;
  queuedJobs: number;
  activeJobs: number;
  storageGb: number;
}

export interface ClipGroup {
  id: string;
  name: string;
  color: string;
  clipCount: number;
  lastUpdated: string;
  rules: string[];
}

export interface IndexJob {
  id: string;
  folder: string;
  status: "queued" | "running" | "complete" | "failed";
  progress: number;
  clipsFound: number;
  startedAt: string;
}

export interface SearchResult {
  id: string;
  title: string;
  path: string;
  game: string;
  timestamp?: string;
  bestTimestamp?: number | null;
  segmentStart?: number | null;
  segmentEnd?: number | null;
  durationSec: number;
  score: number;
  tags: string[];
  summary: string;
  thumbnail?: string;
  matchedModality?: string;
  matchedReason?: string;
  segmentRange?: string;
  activeWeapon?: string | null;
  detectedLoadout?: string[];
  killedByWeapon?: string | null;
  killerName?: string | null;
  deathStatus?: string | null;
}

export interface ClipDetail extends SearchResult {
  transcript: string[];
  events: Array<{
    time: string;
    label: string;
    confidence: number;
  }>;
  technical: {
    resolution: string;
    fps: number;
    codec: string;
    sizeMb: number;
  };
  hudDetections?: Array<{
    slot_key: string;
    is_active: number;
    entity_name?: string | null;
    entity_type?: string | null;
    confidence: number;
  }>;
  deathDetections?: Array<{
    status?: string | null;
    killed_with?: string | null;
    killer_name?: string | null;
    confidence: number;
  }>;
}

export interface LogEntry {
  id: string;
  level: "debug" | "info" | "warn" | "error";
  source: string;
  message: string;
  time: string;
}

export interface CommandResult {
  ok: boolean;
  code: number | null;
  stdout: string;
  stderr: string;
}

export interface RuntimeProbe {
  uv_available: boolean;
  uv_version: string;
  python_available: boolean;
  python_version: string;
  ffmpeg_available: boolean;
  ffmpeg_version: string;
  backend_running: boolean;
  project_root: string;
  data_dir: string;
  models_dir: string;
  log_file: string;
}
