import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../lib/api";
import { useSession } from "../context/SessionContext";
import { Badge, Card } from "../components/ui/primitives";
import { PrivacyStatus } from "../components/PrivacyStatus";

export function DashboardPage() {
  const { session } = useSession();
  const [privacy, setPrivacy] = useState<Awaited<ReturnType<typeof api.privacy>> | null>(null);
  const [health, setHealth] = useState<string>("Checking…");

  useEffect(() => {
    api
      .health()
      .then((h) => setHealth(h.status === "ok" ? "Backend reachable" : h.status))
      .catch(() => setHealth("Backend offline — UI still usable"));
    api.privacy().then(setPrivacy).catch(() => setPrivacy(null));
  }, []);

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-xl font-semibold">Clinical research workstation</h1>
        <p className="text-sm text-muted">
          Signed in as {session.name}. Open a case to review a chest X-ray. Research metrics remain “Not evaluated” until experiments run.
        </p>
      </div>
      <div className="grid gap-4 md:grid-cols-3">
        <Card className="p-4">
          <p className="text-xs uppercase tracking-wide text-muted">System</p>
          <p className="mt-2 text-sm">{health}</p>
          <Badge tone="warn">Phase 0 APIs + full UI</Badge>
        </Card>
        <Card className="p-4">
          <p className="text-xs uppercase tracking-wide text-muted">Local cases</p>
          <p className="mt-2 font-mono text-2xl">{session.cases.length}</p>
          <Link to="/analyze" className="text-sm text-accent">
            New chest X-ray analysis
          </Link>
        </Card>
        <Card className="p-4">
          <p className="text-xs uppercase tracking-wide text-muted">Experiments</p>
          <p className="mt-2 font-mono text-2xl">Not evaluated</p>
          <Link to="/experiments" className="text-sm text-accent">
            Open matrix
          </Link>
        </Card>
      </div>
      <div className="grid gap-4 lg:grid-cols-2">
        <Card className="p-4">
          <h2 className="mb-3 text-sm font-semibold">Privacy protected</h2>
          <PrivacyStatus
            dpEnabled={privacy?.differential_privacy_enabled ?? false}
            epsilon={privacy?.dp_epsilon_target ?? 8}
            delta={privacy?.dp_delta ?? 1e-5}
            encrypted={window.location.protocol === "https:"}
          />
        </Card>
        <Card className="p-4">
          <h2 className="mb-3 text-sm font-semibold">Workflow</h2>
          <ol className="list-decimal space-y-2 pl-4 text-sm text-muted">
            <li>Upload a chest X-ray (stays in this browser session)</li>
            <li>Inspect with windowing, zoom, and overlay tools</li>
            <li>When inference is connected, review class, calibration, and XAI</li>
            <li>Ask grounded questions; generate a report for clinician review</li>
          </ol>
        </Card>
      </div>
    </div>
  );
}
