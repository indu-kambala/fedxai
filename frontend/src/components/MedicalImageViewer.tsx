
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

export function MedicalImageViewer({
  src,
  mode,
  brightness,
  contrast,
  heatmapOpacity,
  overlay,
}: Props) {
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const drag = useRef<{ x: number; y: number } | null>(null);

  useEffect(() => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
    drag.current = null;
  }, [src]);

  function onWheel(event: WheelEvent<HTMLDivElement>) {
    setZoom((z) =>
      Math.min(4, Math.max(0.5, z - event.deltaY * 0.001))
    );
  }

  function onPointerDown(event: ReactPointerEvent<HTMLDivElement>) {
    if (event.button !== 0) return;

    drag.current = {
      x: event.clientX - pan.x,
      y: event.clientY - pan.y,
    };

    event.currentTarget.setPointerCapture(event.pointerId);
  }

  function onPointerMove(event: ReactPointerEvent<HTMLDivElement>) {
    if (!drag.current) return;

    setPan({
      x: event.clientX - drag.current.x,
      y: event.clientY - drag.current.y,
    });
  }

  function onPointerUp(event: ReactPointerEvent<HTMLDivElement>) {
    drag.current = null;

    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  }

  const filter = `brightness(${brightness}) contrast(${contrast})`;

  const showTint =
    overlay &&
    (mode === "overlay" || mode === "heatmap" || mode === "attention");

  if (!src) {
    return (
      <div className="flex h-full min-h-0 w-full items-center justify-center overflow-hidden border border-dashed border-line bg-canvas text-sm text-muted">
        Upload a chest X-ray to open the viewer
      </div>
    );
  }

  return (
    <div
      className="relative h-full min-h-0 w-full touch-pan-y overflow-hidden border border-line bg-black"
      onWheel={onWheel}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerCancel={onPointerUp}
    >
      <div
        className={cn(
          "absolute inset-0 flex min-h-0 min-w-0 items-center justify-center",
          mode === "comparison" && "grid grid-cols-2"
        )}
        style={{
          transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
          transformOrigin: "center center",
        }}
      >
        <img
          src={src}
          alt="Chest X-ray"
          className="block max-h-full min-h-0 max-w-full min-w-0 select-none object-contain"
          style={{ filter }}
          draggable={false}
        />

        {mode === "comparison" && (
          <img
            src={src}
            alt="Comparison view"
            className="block max-h-full min-h-0 max-w-full min-w-0 select-none object-contain opacity-90"
            draggable={false}
          />
        )}
      </div>

      {showTint && (
        <div
          className="pointer-events-none absolute inset-0"
          style={{
            opacity: heatmapOpacity,
            background:
              mode === "attention"
                ? "radial-gradient(circle at 45% 55%, rgba(61,126,166,0.45), transparent 55%)"
                : "radial-gradient(circle at 50% 58%, rgba(196,80,60,0.35), transparent 52%)",
            mixBlendMode: "multiply",
          }}
        />
      )}

      {showTint && (
        <p className="pointer-events-none absolute bottom-2 left-2 max-w-[90%] rounded bg-black/80 px-2 py-1 text-[10px] text-slate-200">
          Illustrative overlay only. This is not a computed clinical attribution map.
        </p>
      )}

      <div className="pointer-events-none absolute right-2 top-2 rounded bg-black/70 px-2 py-1 font-mono text-[10px] text-slate-200">
        {Math.round(zoom * 100)}% · Drag to pan · Scroll to zoom
      </div>
    </div>
  );
}
