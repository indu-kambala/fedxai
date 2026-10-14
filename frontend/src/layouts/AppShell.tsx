import { NavLink, Outlet, useNavigate } from "react-router-dom";
import {
  Activity,
  BookOpen,
  ClipboardList,
  Cpu,
  FileText,
  FlaskConical,
  LayoutDashboard,
  Lock,
  MessageSquare,
  LogOut,
  Moon,
  Network,
  ScanEye,
  Settings,
  Shield,
  Stethoscope,
  Sun,
  Upload,
} from "lucide-react";
import { useSession } from "../context/SessionContext";
import { useTheme } from "../context/ThemeContext";
import { HOSPITALS, MODELS } from "../lib/types";
import { Badge, Select } from "../components/ui/primitives";
import { cn } from "../lib/cn";
import { useEffect, useState } from "react";
import { api } from "../lib/api";

const NAV = [
  { to: "/dashboard", label: "Overview", icon: LayoutDashboard },
  { to: "/analyze", label: "Patient Analysis", icon: Upload },
  { to: "/chat", label: "Clinical Chat", icon: MessageSquare },
  { to: "/cases", label: "Previous Cases", icon: ClipboardList },
  { to: "/reports", label: "Reports", icon: FileText },
  { to: "/knowledge", label: "Knowledge Base", icon: BookOpen },
  { to: "/federated", label: "Federated Learning", icon: Network },
  { to: "/experiments", label: "Experiments", icon: FlaskConical },
  { to: "/models", label: "Model Performance", icon: Cpu },
  { to: "/explainability", label: "Explainability", icon: ScanEye },
  { to: "/privacy", label: "Privacy Center", icon: Shield },
  { to: "/audit", label: "Audit Logs", icon: Lock },
  { to: "/settings", label: "Settings", icon: Settings },
];

export function AppShell() {
  const { session, setHospital, setModel, signOut } = useSession();
  const { theme, toggle } = useTheme();
  const navigate = useNavigate();
  const [apiUp, setApiUp] = useState<boolean | null>(null);

  useEffect(() => {
    api
      .health()
      .then(() => setApiUp(true))
      .catch(() => setApiUp(false));
  }, []);

  return (
    <div className="flex min-h-screen bg-bg text-fg">
      <aside className="flex w-60 shrink-0 flex-col border-r border-line bg-panel">
        <div className="border-b border-line px-4 py-4">
          <p className="flex items-center gap-2 font-semibold">
            <Stethoscope size={16} className="text-accent" />
            FedXAI-Med
          </p>
          <p className="mt-1 text-[11px] leading-snug text-muted">Federated Clinical Intelligence</p>
        </div>
        <nav className="flex-1 space-y-0.5 overflow-y-auto p-2">
          {NAV.map((item) => {
            const Icon = item.icon;
            return (
              <NavLink
                key={item.to}
                to={item.to}
                className={({ isActive }) =>
                  cn(
                    "flex items-center gap-2 rounded-md px-3 py-2 text-sm text-muted hover:bg-panel-hover hover:text-fg",
                    isActive && "bg-panel-hover text-fg"
                  )
                }
              >
                <Icon size={16} />
                {item.label}
              </NavLink>
            );
          })}
        </nav>
      </aside>
      <div className="flex min-w-0 flex-1 flex-col">
        <header className="flex h-14 items-center justify-between gap-4 border-b border-line bg-panel px-4">
          <div className="flex min-w-0 items-center gap-3">
            <Activity size={16} className="text-accent" />
            <span className="truncate text-sm text-muted">Workstation</span>
            <Select value={session.hospitalId} onChange={(e) => setHospital(e.target.value)}>
              {HOSPITALS.map((h) => (
                <option key={h.id} value={h.id}>
                  {h.name}
                </option>
              ))}
            </Select>
            <Select value={session.modelId} onChange={(e) => setModel(e.target.value)}>
              {MODELS.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.name}
                </option>
              ))}
            </Select>
          </div>
          <div className="flex items-center gap-2">
            <Badge tone={apiUp ? "ok" : apiUp === false ? "warn" : "neutral"}>
              {apiUp ? "API online" : apiUp === false ? "API offline" : "API…"}
            </Badge>
            <Badge tone="ok">Privacy protected</Badge>
            <Badge tone="accent">{session.role.replace("_", " ")}</Badge>
            <span className="text-sm">{session.name}</span>
            <button type="button" className="rounded p-1.5 hover:bg-panel-hover" onClick={toggle} aria-label="Toggle theme">
              {theme === "dark" ? <Sun size={16} /> : <Moon size={16} />}
            </button>
            <button
              type="button"
              className="rounded p-1.5 hover:bg-panel-hover"
              onClick={() => {
                signOut();
                navigate("/login");
              }}
              aria-label="Sign out"
            >
              <LogOut size={16} />
            </button>
          </div>
        </header>
        <div className="border-b border-warn/30 bg-warn/10 px-4 py-1.5 text-[11px] text-muted">
          AI-assisted decision support. Outputs require review by a qualified clinician and are not a medical diagnosis.
        </div>
        <main className="min-h-0 flex-1 overflow-y-auto p-4">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
