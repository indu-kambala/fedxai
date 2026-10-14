import { useMemo, useState } from "react";
import { useSession } from "../context/SessionContext";
import { MODELS } from "../lib/types";
import type { CaseRecord } from "../lib/types";
import { MedicalImageViewer } from "../components/MedicalImageViewer";
import { PredictionCard } from "../components/PredictionCard";
import { ConfidenceCard } from "../components/ConfidenceCard";
import { ExplanationPanel } from "../components/ExplanationPanel";
import { HeatmapViewer } from "../components/HeatmapViewer";
import { ClinicalChat } from "../components/ClinicalChat";
import { Button } from "../components/ui/primitives";

type Mode = "original" | "heatmap" | "overlay" | "attention" | "comparison";

export function AnalyzePage({ existing }: { existing?: CaseRecord | null }) {
  const { session, addCase, activeCase } = useSession();
  const current = existing ?? activeCase;
  const [mode, setMode] = useState<Mode>("original");
  const [brightness, setBrightness] = useState(1);
  const [contrast, setContrast] = useState(1);
  const [opacity, setOpacity] = useState(0.35);
  const [overlay, setOverlay] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const model = MODELS.find((m) => m.id === session.modelId) ?? MODELS[2];

  const src = current?.imageDataUrl ?? null;

  function onFile(file: File) {
    setError(null);
    const allowed = ["image/png", "image/jpeg", "image/jpg"];
    if (!allowed.includes(file.type) && !file.name.toLowerCase().endsWith(".dcm")) {
      setError("Allowed types: PNG, JPEG, DICOM (.dcm).");
      return;
    }
    if (file.size > 20 * 1024 * 1024) {
      setError("File exceeds 20 MB upload limit.");
      return;
    }
    if (file.name.toLowerCase().endsWith(".dcm")) {
      setError("DICOM decoding is enabled in preprocessing (Phase 2). Convert to PNG/JPEG for this viewer, or wait for the DICOM pipeline.");
      return;
    }
    const reader = new FileReader();
    reader.onload = () => {
      const record: CaseRecord = {
        id: crypto.randomUUID(),
        createdAt: new Date().toISOString(),
        hospitalId: session.hospitalId,
        modelId: session.modelId,
        fileName: file.name,
        imageDataUrl: String(reader.result),
        predictedClass: null,
        modelProbability: null,
        calibrationStatus: "Requires review",
        uncertainty: "Low confidence / possible out-of-distribution image cannot be assessed until the model is connected.",
        notes: "",
      };
      addCase(record);
    };
    reader.readAsDataURL(file);
  }

  const modes: Mode[] = useMemo(() => ["original", "heatmap", "overlay", "attention", "comparison"], []);

  function downloadReport() {
    if (!current) return;
    const text = [
      "FedXAI-Med case report (research prototype)",
      `Case ID: ${current.id}`,
      `Hospital: ${current.hospitalId}`,
      `Model: ${model.name}`,
      `File: ${current.fileName}`,
      `Prediction: ${current.predictedClass ?? "Not available"}`,
      `Model probability: ${current.modelProbability ?? "Not available"}`,
      `Calibration: ${current.calibrationStatus}`,
      current.uncertainty,
      "This is not a medical diagnosis. Doctor review required.",
      `Timestamp: ${current.createdAt}`,
    ].join("\n");
    const blob = new Blob([text], { type: "text/plain" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `fedxai-report-${current.id}.txt`;
    a.click();
  }

  return (
    <div className="flex h-[calc(100vh-7.5rem)] min-h-[640px] flex-col gap-3">
      <div className="grid min-h-0 flex-1 grid-cols-1 gap-3 xl:grid-cols-[minmax(0,1fr)_320px]">
        <section className="flex min-h-0 flex-col gap-2">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div>
              <h1 className="text-lg font-semibold">Image analysis workspace</h1>
              <p className="text-xs text-muted">{current ? current.fileName : "Upload a chest X-ray — image stays in this session, not on the FL aggregator."}</p>
            </div>
            <label className="inline-flex cursor-pointer items-center rounded-md bg-accent px-3 py-1.5 text-sm text-white">
              Upload chest X-ray
              <input
                type="file"
                accept=".png,.jpg,.jpeg,.dcm,.dicom,image/png,image/jpeg"
                className="hidden"
                onChange={(e) => {
                  const file = e.target.files?.[0];
                  if (file) onFile(file);
                }}
              />
            </label>
          </div>
          {error ? <p className="text-xs text-danger">{error}</p> : null}
          <div className="flex flex-wrap gap-1">
            {modes.map((m) => (
              <button
                key={m}
                type="button"
                onClick={() => setMode(m)}
                className={`rounded border px-2 py-1 text-xs capitalize ${mode === m ? "border-accent text-fg" : "border-line text-muted"}`}
              >
                {m}
              </button>
            ))}
          </div>
          <div className="min-h-0 flex-1">
            <MedicalImageViewer
              src={src}
              mode={mode}
              brightness={brightness}
              contrast={contrast}
              heatmapOpacity={opacity}
              overlay={overlay}
            />
          </div>
          <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
            <label className="text-[11px] text-muted">
              Brightness {brightness.toFixed(2)}
              <input type="range" min={0.4} max={2} step={0.01} value={brightness} onChange={(e) => setBrightness(Number(e.target.value))} className="w-full" />
            </label>
            <label className="text-[11px] text-muted">
              Contrast {contrast.toFixed(2)}
              <input type="range" min={0.4} max={2} step={0.01} value={contrast} onChange={(e) => setContrast(Number(e.target.value))} className="w-full" />
            </label>
            <label className="text-[11px] text-muted">
              Heatmap opacity {opacity.toFixed(2)}
              <input type="range" min={0} max={1} step={0.01} value={opacity} onChange={(e) => setOpacity(Number(e.target.value))} className="w-full" />
            </label>
            <div className="flex items-end gap-2">
              <Button
                variant="outline"
                type="button"
                onClick={() => {
                  setBrightness(1);
                  setContrast(1);
                  setOpacity(0.35);
                  setMode("original");
                }}
              >
                Reset
              </Button>
              <Button variant="ghost" type="button" onClick={() => setOverlay((v) => !v)}>
                Overlay {overlay ? "on" : "off"}
              </Button>
            </div>
          </div>
          <HeatmapViewer enabled={overlay && mode !== "original"} />
        </section>
        <aside className="flex min-h-0 flex-col gap-3 overflow-y-auto">
          <p className="text-xs font-semibold uppercase tracking-wide text-muted">AI clinical analysis</p>
          <PredictionCard predictedClass={current?.predictedClass ?? null} probability={current?.modelProbability ?? null} modelName={model.name} />
          <ConfidenceCard probability={current?.modelProbability ?? null} calibrationStatus={current?.calibrationStatus ?? "Requires review"} />
          <ExplanationPanel predictedClass={current?.predictedClass ?? null} modelFamily={model.family} />
          <div className="rounded-lg border border-line bg-panel p-4 text-sm">
            <h3 className="text-sm font-semibold">Uncertainty</h3>
            <p className="mt-2 text-xs leading-relaxed text-muted">
              {current?.uncertainty ?? "Upload an image. Out-of-distribution checks run after the model is connected. Low-confidence images will be flagged rather than forced into a class."}
            </p>
            <p className="mt-2 text-xs text-muted">Recommended review: independent radiologist/clinician interpretation.</p>
          </div>
          <Button variant="outline" type="button" onClick={downloadReport} disabled={!current}>
            Download explanation report
          </Button>
        </aside>
      </div>
      <section className="h-[280px] shrink-0 border-t border-line pt-2">
        <ClinicalChat compact />
      </section>
    </div>
  );
}
