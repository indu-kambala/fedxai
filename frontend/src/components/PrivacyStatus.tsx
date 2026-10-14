import { Badge } from "./ui/primitives";

export function PrivacyStatus({
  dpEnabled,
  epsilon,
  delta,
  encrypted,
}: {
  dpEnabled: boolean;
  epsilon: number;
  delta: number;
  encrypted: boolean;
}) {
  return (
    <div className="space-y-2 text-sm">
      <div className="flex items-center justify-between">
        <span>Raw images remain at hospital</span>
        <Badge tone="ok">Yes</Badge>
      </div>
      <div className="flex items-center justify-between">
        <span>Federated training</span>
        <Badge tone="ok">Enabled</Badge>
      </div>
      <div className="flex items-center justify-between">
        <span>Differential privacy</span>
        <Badge tone={dpEnabled ? "ok" : "warn"}>{dpEnabled ? "On" : "Off"}</Badge>
      </div>
      <div className="flex items-center justify-between font-mono text-xs text-muted">
        <span>ε / δ</span>
        <span>
          {epsilon} / {delta}
        </span>
      </div>
      <div className="flex items-center justify-between">
        <span>Encrypted communication (prod TLS)</span>
        <Badge tone={encrypted ? "ok" : "warn"}>{encrypted ? "Ready" : "Dev HTTP"}</Badge>
      </div>
      <p className="text-[11px] text-muted">Differential privacy is a tradeoff, not perfect privacy.</p>
    </div>
  );
}
