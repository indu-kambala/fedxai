import { Card } from "./ui/primitives";

export function CitationCard({
  title,
  source,
  excerpt,
}: {
  title: string;
  source: string;
  excerpt: string;
}) {
  return (
    <Card className="p-3">
      <p className="text-xs font-semibold">{title}</p>
      <p className="mt-0.5 font-mono text-[10px] uppercase text-accent">{source}</p>
      <p className="mt-2 text-xs leading-relaxed text-muted">{excerpt}</p>
    </Card>
  );
}
