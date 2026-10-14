import { Card } from "./ui/primitives";

export function ExperimentCard({ id, name, status }: { id: string; name: string; status: string }) {
  const label = status === "not_evaluated" || status === "Not evaluated" ? "Not evaluated" : status;
  return (
    <Card className="p-4">
      <p className="font-mono text-[11px] text-accent">{id}</p>
      <h3 className="mt-1 text-sm font-semibold">{name}</h3>
      <p className="mt-3 font-mono text-xs text-muted">{label}</p>
    </Card>
  );
}
