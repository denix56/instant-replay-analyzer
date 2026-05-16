import { invoke } from "@tauri-apps/api/core";
import type { CommandResult, ModelTier, RuntimeProbe, RuntimeProfile } from "../types";

function isTauriRuntime() {
  return Boolean("__TAURI_INTERNALS__" in window || "__TAURI__" in window);
}

function requireTauri() {
  if (!isTauriRuntime()) {
    throw new Error("Native command unavailable in browser preview");
  }
}

export async function selectFolder() {
  requireTauri();
  return invoke<string | null>("select_folder");
}

export async function startNativeRuntime(config: {
  clipsDir?: string;
  runtimeProfile?: RuntimeProfile;
  modelTier?: ModelTier;
  backendPort?: number;
}) {
  requireTauri();
  return invoke<CommandResult>("start_native_runtime", { config });
}

export async function stopNativeRuntime() {
  requireTauri();
  return invoke<CommandResult>("stop_native_runtime");
}

export async function probeRuntime() {
  requireTauri();
  return invoke<RuntimeProbe>("probe_runtime");
}

export async function suggestedFolders() {
  requireTauri();
  return invoke<string[]>("suggested_folders");
}

export function nativeRuntimeLabel() {
  return isTauriRuntime() ? "Tauri" : "Browser preview";
}
