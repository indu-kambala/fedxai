import { DISCLAIMER } from "../lib/types";
import { Badge, Card } from "./ui/primitives";

type Props = {
  predictedClass: string | null;
  probability: number | null;
  modelName: string;
};

export function PredictionCard({ predictedClass, probability, modelName }: Props) {
  return (
    <Card className="p-4">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold">Prediction</h3>
        <Badge tone={predictedClass ? "accent" : "neutral"}>{predictedClass ? "Model output" : "Pending"}</Badge>
      </div>
      <p className="mt-3 font-mono text-2xl tracking-tight">{predictedClass ?? "Not available"}</p>
      <dl className="mt-3 grid grid-cols-2 gap-2 text-xs text-muted">
        <div>
          <dt>Model probability</dt>
          <dd className="text-fg">{probability == null ? "—" : `${(probability * 100).toFixed(1)}%`}</dd>
        </div>
        <div>
          <dt>Model</dt>
          <dd className="text-fg">{modelName}</dd>
        </div>
      </dl>
      <p className="mt-3 text-[11px] leading-relaxed text-muted">{DISCLAIMER}</p>
    </Card>
  );
}
