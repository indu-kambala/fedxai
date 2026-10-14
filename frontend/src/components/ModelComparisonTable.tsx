import { MODELS } from "../lib/types";

const METRICS = ["Accuracy", "Precision", "Recall", "Macro F1", "ROC-AUC", "Sensitivity", "Specificity", "Calibration", "Latency"];

export function ModelComparisonTable() {
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[720px] text-left text-sm">
        <thead className="border-b border-line text-xs uppercase tracking-wide text-muted">
          <tr>
            <th className="py-2 pr-3">Model</th>
            {METRICS.map((m) => (
              <th key={m} className="py-2 pr-3 font-medium">
                {m}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {MODELS.map((model) => (
            <tr key={model.id} className="border-b border-line/70">
              <td className="py-2 pr-3 font-medium">{model.name}</td>
              {METRICS.map((m) => (
                <td key={m} className="py-2 pr-3 font-mono text-xs text-muted">
                  Not evaluated
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
