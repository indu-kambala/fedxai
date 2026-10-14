export function AuditTimeline({ events }: { events: Array<{ time: string; actor: string; action: string }> }) {
  if (!events.length) {
    return <p className="text-sm text-muted">No persisted audit events yet. Access will be logged when authentication is wired.</p>;
  }
  return (
    <ol className="space-y-3 border-l border-line pl-4">
      {events.map((e) => (
        <li key={e.time + e.action}>
          <p className="font-mono text-[11px] text-muted">{e.time}</p>
          <p className="text-sm">
            <span className="text-muted">{e.actor}</span> — {e.action}
          </p>
        </li>
      ))}
    </ol>
  );
}
