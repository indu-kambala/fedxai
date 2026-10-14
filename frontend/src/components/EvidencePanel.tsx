import { CitationCard } from "./CitationCard";

export function EvidencePanel({
  items,
}: {
  items: Array<{ title: string; source: string; excerpt: string }>;
}) {
  if (!items.length) {
    return <p className="text-xs text-muted">No retrieved documents. RAG is not connected; sources will not be fabricated.</p>;
  }
  return (
    <div className="space-y-2">
      {items.map((item) => (
        <CitationCard key={item.title} {...item} />
      ))}
    </div>
  );
}
