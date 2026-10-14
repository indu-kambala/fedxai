import { Card } from "./ui/primitives";

export function ExplanationPanel({
  predictedClass,
  modelFamily,
}: {
  predictedClass: string | null;
  modelFamily: string;
}) {
  const method =
    modelFamily === "cnn"
      ? "Grad-CAM / Grad-CAM++ will be used for CNN backbones."
      : "Attention rollout / transformer attribution will be used for ViT and Swin. Grad-CAM is not applied blindly.";

  return (
    <Card className="p-4">
      <h3 className="text-sm font-semibold">Model explanation</h3>
      <p className="mt-2 text-sm leading-relaxed text-muted">
        {predictedClass
          ? `The model assigned class ${predictedClass}. Highlighted regions, when present, represent model attribution and should not be interpreted as a definitive clinical finding.`
          : "No explanation is available until a model prediction exists. Heatmaps will not be invented."}
      </p>
      <p className="mt-3 text-[11px] text-muted">{method}</p>
    </Card>
  );
}
