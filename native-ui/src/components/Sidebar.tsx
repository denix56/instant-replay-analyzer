import {
  Activity,
  Cog,
  Database,
  FolderKanban,
  LayoutDashboard,
  ListTree,
  Logs,
  Search,
  SlidersHorizontal,
  Video,
} from "lucide-react";
import type { ViewId } from "../types";

const navItems: Array<{ id: ViewId; label: string; icon: typeof LayoutDashboard }> = [
  { id: "dashboard", label: "Dashboard", icon: LayoutDashboard },
  { id: "setup", label: "First Run", icon: SlidersHorizontal },
  { id: "groups", label: "Groups", icon: FolderKanban },
  { id: "indexing", label: "Indexing", icon: Database },
  { id: "search", label: "Search", icon: Search },
  { id: "clip", label: "Clip Detail", icon: Video },
  { id: "settings", label: "Settings", icon: Cog },
  { id: "logs", label: "Logs/Status", icon: Logs },
];

type SidebarProps = {
  activeView: ViewId;
  onNavigate: (view: ViewId) => void;
  backendLabel: string;
  setupComplete: boolean;
};

export function Sidebar({ activeView, onNavigate, backendLabel, setupComplete }: SidebarProps) {
  return (
    <aside className="sidebar">
      <div className="brand-block">
        <Activity size={20} />
        <div>
          <div className="brand-title">Instant Replay</div>
          <div className="brand-subtitle">Semantic clip search</div>
        </div>
      </div>

      <nav className="nav-list" aria-label="Primary">
        {navItems.map((item) => {
          const Icon = item.icon;
          const selected = item.id === activeView;
          return (
            <button
              key={item.id}
              type="button"
              className={selected ? "nav-item nav-item-active" : "nav-item"}
              onClick={() => onNavigate(item.id)}
            >
              <Icon size={17} />
              <span>{item.label}</span>
            </button>
          );
        })}
      </nav>

      <div className="sidebar-footer">
        <div className="metric-row">
          <ListTree size={15} />
          <span>{setupComplete ? "Setup complete" : "Setup pending"}</span>
        </div>
        <div className="metric-row metric-muted">{backendLabel}</div>
      </div>
    </aside>
  );
}
