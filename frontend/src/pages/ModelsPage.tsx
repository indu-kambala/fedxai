import { ModelComparisonTable } from "../components/ModelComparisonTable";
import { ConfusionMatrix } from "../components/ConfusionMatrix";
import { Card } from "../components/ui/primitives";

export function ModelsPage() {
  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-xl font-semibold">Model performance</h1>
        <p className="text-sm text-muted">
          DenseNet-121, EfficientNet-B3, ViT-B/16, and Swin are configured. Metrics are filled only from measured runs. Selection uses macro-F1, ROC-AUC, sensitivity, specificity, calibration, and cost — not accuracy alone.
        </p>
      </div>
      <Card className="p-4">
        <ModelComparisonTable />
      </Card>
      <div className="grid gap-4 lg:grid-cols-2">
        <Card className="p-4">
          <h2 className="mb-3 text-sm font-semibold">Confusion matrix</h2>
          <ConfusionMatrix />
        </Card>
        <Card className="p-4">
          <h2 className="mb-3 text-sm font-semibold">Curves</h2>
          <p className="text-sm text-muted">ROC, PR, training, and calibration curves: Not evaluated.</p>
        </Card>
      </div>
    </div>
  );
}
