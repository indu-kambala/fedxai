import { Card } from "../components/ui/primitives";

export function ExplainabilityPage() {
  return (
    <div className="space-y-4">
      <h1 className="text-xl font-semibold">Explainability</h1>
      <p className="text-sm text-muted">
        Attribution is architecture-specific. Heatmaps are not clinical findings.
      </p>
      <div className="grid gap-4 md:grid-cols-2">
        <Card className="p-4">
          <h2 className="text-sm font-semibold">CNN (DenseNet, EfficientNet)</h2>
          <ul className="mt-2 list-disc pl-4 text-sm text-muted">
            <li>Grad-CAM</li>
            <li>Grad-CAM++</li>
            <li>Optional SHAP</li>
          </ul>
        </Card>
        <Card className="p-4">
          <h2 className="text-sm font-semibold">ViT / Swin</h2>
          <ul className="mt-2 list-disc pl-4 text-sm text-muted">
            <li>Attention rollout / attention maps</li>
            <li>Transformer-compatible attribution</li>
            <li>Grad-CAM-style methods only when technically valid</li>
          </ul>
        </Card>
      </div>
      <Card className="p-4 text-sm text-muted">
        Engine status: Not evaluated. Open a case in the analysis workspace to use the overlay viewer. Computed maps will appear here after Phase 10.
      </Card>
    </div>
  );
}
