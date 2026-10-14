import { useEffect, useRef, useState, type PointerEvent as ReactPointerEvent, type WheelEvent } from "react";
import { cn } from "../lib/cn";

type Mode = "original" | "heatmap" | "overlay" | "attention" | "comparison";

type Props = {
  src: string | null;
  mode: Mode;
  brightness: number;
  contrast: number;
  heatmapOpacity: number;
  overlay: boolean;
};

export function MedicalImageViewer({ src, mode, brightness, contrast, heatmapOpacity, overlay }: Props) {
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const drag = useRef<{ x: number; y: number } | null>(null);

  useEffect(() => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  }, [src]);

  function onWheel(event: WheelEvent) {
    event.preventDefault();
    setZoom((z) => Math.min(6, Math.max(0.5, z - event.deltaY * 0.001)));
  }

  function onPointerDown(event: ReactPointerEvent) {
    drag.current = { x: event.clientX - pan.x, y: event.clientY - pan.y };
    (event.target as HTMLElement).setPointerCapture(event.pointerId);
  }

  function onPointerMove(event: ReactPointerEvent) {
    if (!drag.current) return;
    setPan({ x: event.clientX - drag.current.x, y: event.clientY - drag.current.y });
  }

  function onPointerUp() {
    drag.current = null;
  }

  if (!src) {
    return (
      <div className="flex h-full min-h-[420px] items-center justify-center border border-dashed border-line bg-canvas text-sm text-muted">
        Upload a chest X-ray to open the viewer
      </div>
    );
  }

  const filter = `brightness(${brightness}) contrast(${contrast})`;
  const showTint = overlay && (mode === "overlay" || mode === "heatmap" || mode === "attention");

  return (
    <div
      className="relative h-full min-h-[420px] overflow-hidden border border-line bg-black"
      onWheel={onWheel}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
    >
      <div
        className={cn("absolute inset-0 flex items-center justify-center", mode === "comparison" && "grid grid-cols-2")}
        style={{ transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})` }}
      >
        <img src={src} alt="Chest X-ray" className="max-h-full max-w-full select-none object-contain" style={{ filter }} draggable={false} />
        {mode === "comparison" ? (
          <img src={src} alt="Comparison pane" className="max-h-full max-w-full select-none object-contain opacity-90" draggable={false} />
        ) : null}
      </div>
      {showTint ? (
        <div
          className="pointer-events-none absolute inset-0 mix-blend-multiply"
          style={{
            opacity: heatmapOpacity,
            background:
              mode === "attention"
                ? "radial-gradient(circle at 45% 55%, rgba(61,126,166,0.45), transparent 55%)"
                : "radial-gradient(circle at 50% 58%, rgba(196,80,60,0.35), transparent 52%)",
          }}
        />
      ) : null}
      {showTint ? (
        <p className="pointer-events-none absolute bottom-3 left-3 max-w-md rounded bg-black/70 px-2 py-1 text-[11px] text-slate-200">
          Attribution overlay is illustrative of the viewer, not a computed Grad-CAM/attention map. No clinical finding is implied.
        </p>
      ) : null}
      <div className="pointer-events-none absolute right-3 top-3 font-mono text-[11px] text-slate-300">
        {Math.round(zoom * 100)}% · drag to pan · scroll to zoom
      </div>
    </div>
  );
}

export function resetViewerHint() {
  return { zoom: 1, pan: { x: 0, y: 0 } };
}
