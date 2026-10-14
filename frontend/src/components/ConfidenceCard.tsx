import { Card } from "./ui/primitives";

export function ConfidenceCard({
  probability,
  calibrationStatus,
}: {
  probability: number | null;
  calibrationStatus: string;
}) {
  return (
    <Card className="p-4">
      <h3 className="text-sm font-semibold">Confidence</h3>
      <p className="mt-2 font-mono text-xl">{probability == null ? "—" : `${(probability * 100).toFixed(1)}%`}</p>
      <p className="mt-2 text-xs text-muted">Calibration status: {calibrationStatus}</p>
      <p className="mt-2 text-[11px] leading-relaxed text-muted">
        AI confidence is not equivalent to diagnostic certainty. Softmax scores are not guaranteed clinical confidence.
      </p>
    </Card>
  );
}
