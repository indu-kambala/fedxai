
import { useMemo, useState } from "react";
import { useSession } from "../context/SessionContext";
import { MODELS } from "../lib/types";
import type { CaseRecord } from "../lib/types";
import { api } from "../lib/api";
import { MedicalImageViewer } from "../components/MedicalImageViewer";
import { PredictionCard } from "../components/PredictionCard";
import { ConfidenceCard } from "../components/ConfidenceCard";
import { ExplanationPanel } from "../components/ExplanationPanel";
import { HeatmapViewer } from "../components/HeatmapViewer";
import { ClinicalChat } from "../components/ClinicalChat";
import { Button } from "../components/ui/primitives";

type Mode = "original" | "heatmap" | "overlay" | "attention" | "comparison";

export function AnalyzePage({
  existing,
}: {
  existing?: CaseRecord | null;
}) {
  const { session, addCase, updateCase, activeCase } = useSession();
  const current = existing ?? activeCase;

  const [mode, setMode] = useState<Mode>("original");
  const [brightness, setBrightness] = useState(1);
  const [contrast, setContrast] = useState(1);
  const [opacity, setOpacity] = useState(0.35);
  const [overlay, setOverlay] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [analyzing, setAnalyzing] = useState(false);

  const model = MODELS.find((m) => m.id === session.modelId) ?? MODELS[2];
  const src = current?.imageDataUrl ?? null;

  const modes: Mode[] = useMemo(
    () => ["original", "heatmap", "overlay", "attention", "comparison"],
    []
  );

  function onFile(file: File) {
    setError(null);

    const allowed = ["image/png", "image/jpeg", "image/jpg"];

    if (
      !allowed.includes(file.type) &&
      !file.name.toLowerCase().endsWith(".dcm")
    ) {
      setError(
        "Allowed types: PNG and JPEG. DICOM is not supported by this viewer yet."
      );
      return;
    }

    if (file.size > 20 * 1024 * 1024) {
      setError("File exceeds the 20 MB upload limit.");
      return;
    }

    if (file.name.toLowerCase().endsWith(".dcm")) {
      setError("Please convert the DICOM image to PNG or JPEG before uploading.");
      return;
    }

    const reader = new FileReader();

    reader.onload = () => {
      if (typeof reader.result !== "string") {
        setError("Could not read this image. Please try another file.");
        return;
      }

      const record: CaseRecord = {
        id: crypto.randomUUID(),
        createdAt: new Date().toISOString(),
        hospitalId: session.hospitalId,
        modelId: session.modelId,
        fileName: file.name,
        imageDataUrl: reader.result,
        predictedClass: null,
        modelProbability: null,
        calibrationStatus: "Requires review",
        uncertainty: "Model inference has not yet been run for this image.",
        notes: "",
      };

      addCase(record);
    };

    reader.onerror = () => {
      setError("Could not read this image. Please try another file.");
    };

    reader.readAsDataURL(file);
  }

  async function analyzeImage() {
    if (!current?.imageDataUrl || analyzing) {
      return;
    }

    setError(null);
    setAnalyzing(true);

    try {
      const imageResponse = await fetch(current.imageDataUrl);

      if (!imageResponse.ok) {
        throw new Error("Could not prepare the selected image for analysis.");
      }

      const imageBlob = await imageResponse.blob();

      const file = new File(
        [imageBlob],
        current.fileName || "chest-xray.png",
        {
          type: imageBlob.type || "image/png",
        }
      );

      const result = await api.analyze(
        file,
        session.hospitalId,
        current.id,
        model.id
      );

      console.log("FedXAI analysis response:", result);

      if (
        typeof result.predicted_class !== "string" ||
        typeof result.model_probability !== "number" ||
        !Number.isFinite(result.model_probability) ||
        result.model_probability < 0 ||
        result.model_probability > 1
      ) {
        throw new Error("The API response did not contain a valid prediction.");
      }

      updateCase(current.id, {
        predictedClass: result.predicted_class,
        modelProbability: result.model_probability,
        calibrationStatus: result.calibration_status || "Requires review",
        uncertainty:
          "Inference completed. The model score is not calibrated clinical certainty. Independent clinician review is required.",
      });
    } catch (err) {
      console.error("FedXAI image analysis failed:", err);

      setError(
        err instanceof Error
          ? err.message
          : "An unexpected error occurred during analysis."
      );
    } finally {
      setAnalyzing(false);
    }
  }

  function downloadReport() {
    if (!current) {
      return;
    }

    const text = [
      "FedXAI-Med case report (research prototype)",
      `Case ID: ${current.id}`,
      `Hospital: ${current.hospitalId}`,
      `Model selected: ${model.name}`,
      `File: ${current.fileName}`,
      `Prediction: ${current.predictedClass ?? "Not available"}`,
      `Model probability: ${
        current.modelProbability == null
          ? "Not available"
          : `${(current.modelProbability * 100).toFixed(2)}%`
      }`,
      `Calibration: ${current.calibrationStatus}`,
      current.uncertainty,
      "This is not a medical diagnosis. Qualified clinician review is required.",
      `Timestamp: ${current.createdAt}`,
    ].join("\n");

    const blob = new Blob([text], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");

    a.href = url;
    a.download = `fedxai-report-${current.id}.txt`;
    document.body.appendChild(a);
    a.click();
    a.remove();

    URL.revokeObjectURL(url);
  }

  return (
    <div className="flex min-h-0 flex-col gap-3">
      <div className="grid min-h-0 grid-cols-1 gap-3 xl:grid-cols-[minmax(0,1fr)_320px]">
        <section className="flex min-w-0 flex-col gap-2">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div>
              <h1 className="text-lg font-semibold">
                Image analysis workspace
              </h1>
              <p className="text-xs text-muted">
                {current
                  ? current.fileName
                  : "Upload a chest X-ray to begin analysis."}
              </p>
            </div>

            <div className="flex flex-wrap items-center gap-2">
              <label className="inline-flex cursor-pointer items-center rounded-md bg-accent px-3 py-2 text-sm text-white hover:opacity-90">
                Upload chest X-ray
                <input
                  type="file"
                  accept=".png,.jpg,.jpeg,image/png,image/jpeg"
                  className="hidden"
                  onChange={(e) => {
                    const file = e.target.files?.[0];

                    if (file) {
                      onFile(file);
                    }

                    e.currentTarget.value = "";
                  }}
                />
              </label>

              <Button
                type="button"
                onClick={() => void analyzeImage()}
                disabled={!current?.imageDataUrl || analyzing}
              >
                {analyzing ? "Analyzing..." : "Analyze X-ray"}
              </Button>
            </div>
          </div>

          {analyzing && (
            <div
              className="rounded-md border border-line bg-panel p-3 text-sm"
              role="status"
            >
              Running model inference. Please wait...
              <div className="mt-2 h-1.5 overflow-hidden rounded bg-line">
                <div className="h-full w-1/2 animate-pulse rounded bg-accent" />
              </div>
            </div>
          )}

          {error && (
            <div
              className="rounded-md border border-danger p-3 text-sm text-danger"
              role="alert"
            >
              {error}
            </div>
          )}

          <div className="flex flex-wrap gap-1">
            {modes.map((m) => (
              <button
                key={m}
                type="button"
                onClick={() => setMode(m)}
                className={`rounded border px-3 py-2 text-xs capitalize ${
                  mode === m
                    ? "border-accent text-fg"
                    : "border-line text-muted"
                }`}
              >
                {m}
              </button>
            ))}
          </div>

          <div className="h-[min(58vh,650px)] min-h-[320px]">
            <MedicalImageViewer
              src={src}
              mode={mode}
              brightness={brightness}
              contrast={contrast}
              heatmapOpacity={opacity}
              overlay={overlay}
            />
          </div>

          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-4">
            <label className="text-xs text-muted">
              Brightness {brightness.toFixed(2)}
              <input
                type="range"
                min={0.4}
                max={2}
                step={0.01}
                value={brightness}
                onChange={(e) => setBrightness(Number(e.target.value))}
                className="w-full"
              />
            </label>

            <label className="text-xs text-muted">
              Contrast {contrast.toFixed(2)}
              <input
                type="range"
                min={0.4}
                max={2}
                step={0.01}
                value={contrast}
                onChange={(e) => setContrast(Number(e.target.value))}
                className="w-full"
              />
            </label>

            <label className="text-xs text-muted">
              Heatmap opacity {opacity.toFixed(2)}
              <input
                type="range"
                min={0}
                max={1}
                step={0.01}
                value={opacity}
                onChange={(e) => setOpacity(Number(e.target.value))}
                className="w-full"
              />
            </label>

            <div className="flex flex-wrap items-end gap-2">
              <Button
                variant="outline"
                type="button"
                onClick={() => {
                  setBrightness(1);
                  setContrast(1);
                  setOpacity(0.35);
                  setMode("original");
                  setOverlay(true);
                }}
              >
                Reset
              </Button>

              <Button
                variant="ghost"
                type="button"
                onClick={() => setOverlay((value) => !value)}
              >
                Overlay {overlay ? "on" : "off"}
              </Button>
            </div>
          </div>

          <HeatmapViewer enabled={overlay && mode !== "original"} />
        </section>

        <aside className="flex min-h-0 flex-col gap-3 xl:max-h-[calc(100vh-7rem)] xl:overflow-y-auto">
          <p className="text-xs font-semibold uppercase tracking-wide text-muted">
            AI clinical analysis
          </p>

          <PredictionCard
            predictedClass={current?.predictedClass ?? null}
            probability={current?.modelProbability ?? null}
            modelName={model.name}
          />

          <ConfidenceCard
            probability={current?.modelProbability ?? null}
            calibrationStatus={
              current?.calibrationStatus ?? "Requires review"
            }
          />

          <ExplanationPanel
            predictedClass={current?.predictedClass ?? null}
            modelFamily={model.family}
          />

          <div className="rounded-lg border border-line bg-panel p-4 text-sm">
            <h3 className="text-sm font-semibold">Uncertainty</h3>
            <p className="mt-2 text-xs leading-relaxed text-muted">
              {current?.uncertainty ??
                "Upload an image and run model inference. The model output requires clinician review."}
            </p>
            <p className="mt-2 text-xs text-muted">
              Model scores are not calibrated diagnostic certainty. Independent
              clinician review is required.
            </p>
          </div>

          <Button
            variant="outline"
            type="button"
            onClick={downloadReport}
            disabled={!current}
          >
            Download explanation report
          </Button>
        </aside>
      </div>

      <section className="min-h-[320px] border-t border-line pt-3">
        <ClinicalChat compact={false} />
      </section>
    </div>
  );
}
