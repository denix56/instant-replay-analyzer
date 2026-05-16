import type {
  BackendHealth,
  ClipDetail,
  ClipGroup,
  DashboardStats,
  IndexJob,
  LogEntry,
  ModelTier,
  RuntimeProfile,
  SearchResult,
} from "../types";

type ApiOptions = {
  backendUrl: string;
};

type SearchParams = {
  query: string;
  groupId?: string;
  minScore?: number;
  game?: string;
};

type BackendSearchResponse = {
  min_score?: number;
  result_count?: number;
  results?: Array<{
    clip_id: number;
    clip_filename: string;
    source_path: string;
    group_name: string;
    best_timestamp?: number | null;
    segment_start?: number | null;
    segment_end?: number | null;
    preview_frame?: string | null;
    summary?: string | null;
    tags?: string[];
    score: number;
    matched_modality?: string;
    matched_reason?: string;
    active_weapon?: string | null;
    detected_loadout?: string[];
    killed_by_weapon?: string | null;
    killer_name?: string | null;
    death_status?: string | null;
  }>;
};

const timeoutMs = 1800;

async function request<T>(backendUrl: string, path: string, init?: RequestInit): Promise<T> {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(`${backendUrl}${path}`, {
      ...init,
      headers: {
        "Content-Type": "application/json",
        ...init?.headers,
      },
      signal: controller.signal,
    });

    if (!response.ok) {
      throw new Error(`${response.status} ${response.statusText}`);
    }

    return (await response.json()) as T;
  } finally {
    window.clearTimeout(timeout);
  }
}

export function createApi({ backendUrl }: ApiOptions) {
  return {
    async health(): Promise<BackendHealth> {
      const started = performance.now();
      try {
        const payload = await request<Partial<BackendHealth>>(backendUrl, "/health");
        return {
          state: payload.state ?? "online",
          version: payload.version,
          message: payload.message ?? "Backend reachable",
          latencyMs: Math.round(performance.now() - started),
        };
      } catch (error) {
        return {
          state: "offline",
          message: error instanceof Error ? error.message : "Backend unavailable",
          latencyMs: Math.round(performance.now() - started),
        };
      }
    },

    dashboard() {
      return request<DashboardStats>(backendUrl, "/api/dashboard");
    },

    groups() {
      return request<ClipGroup[]>(backendUrl, "/api/groups");
    },

    indexJobs() {
      return request<IndexJob[]>(backendUrl, "/api/index/jobs");
    },

    startIndexing(folder: string, profile: RuntimeProfile, tier: ModelTier) {
      return request<IndexJob>(backendUrl, "/api/index/jobs", {
        method: "POST",
        body: JSON.stringify({ folder, runtime_profile: profile, model_tier: tier, input: folder }),
      });
    },

    async search(params: SearchParams) {
      const response = await request<BackendSearchResponse>(backendUrl, "/api/search", {
        method: "POST",
        body: JSON.stringify({
          query: params.query,
          group_name: params.groupId,
          min_score: params.minScore ?? 0.35,
          candidate_limit: 100,
        }),
      });
      return (response.results ?? []).map((item) => {
        const segmentStart = item.segment_start;
        const segmentEnd = item.segment_end;
        const hasSegment = segmentStart != null && segmentEnd != null;
        return {
          id: String(item.clip_id),
          title: item.clip_filename,
          path: item.source_path,
          game: item.group_name,
          timestamp: item.best_timestamp != null ? formatTimestamp(item.best_timestamp) : undefined,
          bestTimestamp: item.best_timestamp,
          segmentStart,
          segmentEnd,
          durationSec: hasSegment ? Math.max(0, Number(segmentEnd) - Number(segmentStart)) : 0,
          score: item.score,
          tags: item.tags ?? [],
          summary: item.summary ?? "",
          thumbnail: item.preview_frame ?? undefined,
          matchedModality: item.matched_modality,
          matchedReason: item.matched_reason,
          activeWeapon: item.active_weapon,
          detectedLoadout: item.detected_loadout ?? [],
          killedByWeapon: item.killed_by_weapon,
          killerName: item.killer_name,
          deathStatus: item.death_status,
          segmentRange: hasSegment ? `${formatTimestamp(segmentStart)}-${formatTimestamp(segmentEnd)}` : undefined,
        };
      });
    },

    async clip(id: string) {
      const item = await request<ClipDetail & {
        active_weapon?: string | null;
        detected_loadout?: string[];
        killed_by_weapon?: string | null;
        killer_name?: string | null;
        death_status?: string | null;
        hud_detections?: ClipDetail["hudDetections"];
        death_detections?: ClipDetail["deathDetections"];
      }>(backendUrl, `/api/clips/${encodeURIComponent(id)}`);
      return {
        ...item,
        activeWeapon: item.activeWeapon ?? item.active_weapon,
        detectedLoadout: item.detectedLoadout ?? item.detected_loadout ?? [],
        killedByWeapon: item.killedByWeapon ?? item.killed_by_weapon,
        killerName: item.killerName ?? item.killer_name,
        deathStatus: item.deathStatus ?? item.death_status,
        hudDetections: item.hudDetections ?? item.hud_detections ?? [],
        deathDetections: item.deathDetections ?? item.death_detections ?? [],
      };
    },

    logs(limit = 200) {
      return request<LogEntry[]>(backendUrl, `/api/logs?limit=${limit}`);
    },

    saveSettings(payload: Record<string, unknown>) {
      return request<{ ok: boolean }>(backendUrl, "/api/settings", {
        method: "PUT",
        body: JSON.stringify(payload),
      });
    },
  };
}

function formatTimestamp(seconds: number) {
  const safe = Math.max(0, Math.floor(seconds));
  const minutes = Math.floor(safe / 60);
  const remaining = safe % 60;
  return `${String(minutes).padStart(2, "0")}:${String(remaining).padStart(2, "0")}`;
}
