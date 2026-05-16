import type { BackendState } from "../types";

type StatusPillProps = {
  state: BackendState | "running" | "complete" | "failed" | "queued" | "warn";
  label: string;
};

export function StatusPill({ state, label }: StatusPillProps) {
  return <span className={`status-pill status-${state}`}>{label}</span>;
}
