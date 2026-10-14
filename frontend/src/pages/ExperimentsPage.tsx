import { useEffect, useState } from "react";
import { api, type ExperimentItem } from "../lib/api";
import { ExperimentCard } from "../components/ExperimentCard";

const FALLBACK: ExperimentItem[] = [
  { id: "exp01", name: "Baseline CNN", status: "not_evaluated" },
  { id: "exp02", name: "Transfer learning", status: "not_evaluated" },
  { id: "exp03", name: "ViT", status: "not_evaluated" },
  { id: "exp04", name: "Swin", status: "not_evaluated" },
  { id: "exp05", name: "Best model + conventional augmentation", status: "not_evaluated" },
  { id: "exp06", name: "Best model + diffusion augmentation", status: "not_evaluated" },
  { id: "exp07", name: "Centralized training", status: "not_evaluated" },
  { id: "exp08", name: "FedAvg", status: "not_evaluated" },
  { id: "exp09", name: "FedProx", status: "not_evaluated" },
  { id: "exp10", name: "FedProx + Differential Privacy", status: "not_evaluated" },
  { id: "exp11", name: "FedProx + DP + diffusion augmentation", status: "not_evaluated" },
];

export function ExperimentsPage() {
  const [items, setItems] = useState<ExperimentItem[]>(FALLBACK);

  useEffect(() => {
    api
      .experiments()
      .then((r) => setItems(r.matrix))
      .catch(() => setItems(FALLBACK));
  }, []);

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-xl font-semibold">Experiment results</h1>
        <p className="text-sm text-muted">Ablation matrix for the paper. Values are never invented.</p>
      </div>
      <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
        {items.map((item) => (
          <ExperimentCard key={item.id} id={item.id} name={item.name} status={item.status} />
        ))}
      </div>
    </div>
  );
}
