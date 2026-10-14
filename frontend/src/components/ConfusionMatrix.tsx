export function ConfusionMatrix() {
  const labels = ["NORMAL", "PNEUMONIA", "COVID-19"];
  return (
    <div>
      <p className="mb-2 text-xs text-muted">Confusion matrix — Not evaluated (no test run stored).</p>
      <div className="grid grid-cols-4 gap-1 text-center font-mono text-xs">
        <div />
        {labels.map((l) => (
          <div key={l} className="text-[10px] text-muted">
            {l}
          </div>
        ))}
        {labels.map((row) => (
          <div key={row} className="contents">
            <div className="flex items-center text-[10px] text-muted">{row}</div>
            {labels.map((col) => (
              <div key={col} className="flex h-12 items-center justify-center border border-line bg-canvas text-muted">
                —
              </div>
            ))}
          </div>
        ))}
      </div>
    </div>
  );
}
