import { useSession } from "../context/SessionContext";
import { MODELS } from "../lib/types";
import { Button, Card } from "../components/ui/primitives";

export function ReportsPage() {
  const { session, activeCase } = useSession();
  const model = MODELS.find((m) => m.id === session.modelId);

  function generate() {
    const c = activeCase;
    const body = [
      "FedXAI-Med AI analysis report",
      "Research prototype — not a confirmed diagnosis",
      `Case ID: ${c?.id ?? "—"}`,
      `Model version: ${model?.name ?? session.modelId}`,
      `Prediction: ${c?.predictedClass ?? "Not available"}`,
      `Confidence: ${c?.modelProbability ?? "Not available"}`,
      "XAI visualization: not computed",
      `Uncertainty: ${c?.uncertainty ?? "—"}`,
      "Retrieved literature: none until RAG is connected",
      "AI limitations: see docs/research/limitations.md",
      `Timestamp: ${new Date().toISOString()}`,
      "Doctor review: ______________________________",
      "Disclaimer: AI-assisted; requires qualified clinician review.",
    ].join("\n");
    const blob = new Blob([body], { type: "text/plain" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "fedxai-case-report.txt";
    a.click();
  }

  return (
    <div className="space-y-4">
      <h1 className="text-xl font-semibold">Reports</h1>
      <Card className="max-w-2xl p-4">
        <p className="text-sm text-muted">
          Professional case report with prediction, explanation, uncertainty, literature, limitations, timestamp, and a doctor review block. Active case: {activeCase?.fileName ?? "none"}.
        </p>
        <Button className="mt-4" type="button" onClick={generate}>
          Generate report
        </Button>
      </Card>
    </div>
  );
}
