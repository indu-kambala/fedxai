import { Badge } from "./ui/primitives";

export function HospitalStatus({ name, status }: { name: string; status: string }) {
  const tone = status === "Complete" ? "ok" : status === "Training" ? "accent" : "neutral";
  return (
    <div className="flex items-center justify-between border-b border-line py-2 text-sm last:border-0">
      <span>{name}</span>
      <Badge tone={tone}>{status}</Badge>
    </div>
  );
}
