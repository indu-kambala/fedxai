export function HeatmapViewer({ enabled }: { enabled: boolean }) {
  return (
    <p className="text-[11px] text-muted">
      {enabled
        ? "Overlay on. This preview tint is a viewer control, not a computed CAM until Phase 10."
        : "Overlay off."}
    </p>
  );
}
