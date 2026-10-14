import { useEffect, useState } from "react";
import { api } from "../lib/api";
import { Card } from "../components/ui/primitives";

export function FederatedRoundsPage() {
  const [status, setStatus] = useState("Loading…");

  useEffect(() => {
    api
      .federatedRounds()
      .then((r) => setStatus(r.status))
      .catch(() => setStatus("Not evaluated"));
  }, []);

  return (
    <div className="space-y-4">
      <h1 className="text-xl font-semibold">Communication rounds</h1>
      <Card className="p-4">
        <p className="text-sm text-muted">
          Per-round log of participation, local loss, global metrics, communication time, update size, and client drift.
        </p>
        <p className="mt-4 font-mono text-sm">{status === "not_evaluated" ? "Not evaluated" : status}</p>
        <p className="mt-2 text-sm text-muted">No rounds stored.</p>
      </Card>
    </div>
  );
}
