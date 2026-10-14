import { useEffect, useState } from "react";
import { api } from "../lib/api";
import { useSession } from "../context/SessionContext";
import { AuditTimeline } from "../components/AuditTimeline";
import { Card } from "../components/ui/primitives";

export function AuditPage() {
  const { session } = useSession();
  const [note, setNote] = useState("");
  const local = [
    {
      time: new Date().toISOString(),
      actor: session.name,
      action: `Session active at ${session.hospitalId} with model ${session.modelId}`,
    },
  ];

  useEffect(() => {
    api
      .audit()
      .then((r) => setNote(r.note))
      .catch(() => setNote("Audit API offline."));
  }, []);

  return (
    <div className="space-y-4">
      <h1 className="text-xl font-semibold">Audit logs</h1>
      <p className="text-sm text-muted">{note}</p>
      <Card className="p-4">
        <AuditTimeline events={local} />
      </Card>
    </div>
  );
}
