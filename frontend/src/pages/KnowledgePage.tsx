import { Card } from "../components/ui/primitives";

const DOCS = [
  {
    title: "System architecture",
    source: "docs/architecture/overview.md",
    excerpt: "Doctor UI, FastAPI, inference, RAG, and five hospital clients. Aggregator receives weights only.",
  },
  {
    title: "Data leakage policy",
    source: "docs/methodology/data_leakage.md",
    excerpt: "Split before augmentation. Patient-level isolation. No synthetic images from the test set.",
  },
  {
    title: "Limitations",
    source: "docs/research/limitations.md",
    excerpt: "Small data, domain shift, LLM hallucination risk, privacy-utility tradeoff, no prospective validation.",
  },
];

export function KnowledgePage() {
  return (
    <div className="space-y-4">
      <h1 className="text-xl font-semibold">Knowledge base</h1>
      <p className="text-sm text-muted">
        RAG will retrieve these sources with citations. Until the vector index is built, only project documentation is listed. PubMed entries will appear only when legally ingested.
      </p>
      <div className="grid gap-3 md:grid-cols-2">
        {DOCS.map((d) => (
          <Card key={d.title} className="p-4">
            <h2 className="text-sm font-semibold">{d.title}</h2>
            <p className="mt-1 font-mono text-[11px] text-accent">{d.source}</p>
            <p className="mt-2 text-sm text-muted">{d.excerpt}</p>
          </Card>
        ))}
      </div>
    </div>
  );
}
