import { useEffect, useRef, useState } from "react";
import { Copy, RotateCcw, ThumbsDown, ThumbsUp, Trash2 } from "lucide-react";
import { useSession } from "../context/SessionContext";
import { groundedReply, suggestedQuestions } from "../lib/chat";
import { MODELS } from "../lib/types";
import { EvidencePanel } from "./EvidencePanel";
import { Button } from "./ui/primitives";

function uid() {
  return crypto.randomUUID();
}

export function ClinicalChat({ compact = false }: { compact?: boolean }) {
  const { session, activeCase, addMessage, clearMessages } = useSession();
  const [draft, setDraft] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [evidenceOpen, setEvidenceOpen] = useState(true);
  const end = useRef<HTMLDivElement>(null);
  const modelName = MODELS.find((m) => m.id === session.modelId)?.name ?? session.modelId;
  const lastCitations = [...session.messages].reverse().find((m) => m.role === "assistant")?.citations ?? [];

  useEffect(() => {
    end.current?.scrollIntoView({ behavior: "smooth" });
  }, [session.messages, streaming]);

  async function send(text: string) {
    const question = text.trim();
    if (!question || streaming) return;
    addMessage({ id: uid(), role: "doctor", content: question, createdAt: new Date().toISOString() });
    setDraft("");
    setStreaming(true);
    const { content, citations } = groundedReply(question, activeCase, modelName);
    let shown = "";
    for (const word of content.split(" ")) {
      shown += (shown ? " " : "") + word;
      await new Promise((r) => setTimeout(r, 18));
      if (shown.split(" ").length % 8 === 0) {
        /* keep UI responsive */
      }
    }
    addMessage({
      id: uid(),
      role: "assistant",
      content,
      createdAt: new Date().toISOString(),
      citations,
    });
    setStreaming(false);
  }

  function exportConversation() {
    const blob = new Blob(
      [session.messages.map((m) => `[${m.createdAt}] ${m.role}: ${m.content}`).join("\n\n")],
      { type: "text/plain" }
    );
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `fedxai-chat-${session.activeCaseId ?? "session"}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  }

  return (
    <div className={compact ? "flex h-full min-h-[240px] flex-col" : "flex h-[calc(100vh-8rem)] flex-col"}>
      <div className="mb-2 flex items-center justify-between gap-2">
        <div>
          <p className="text-sm font-semibold">Doctor ↔ FedXAI-Med</p>
          <p className="text-[11px] text-muted">Grounded clinical decision support. Not a diagnosis.</p>
        </div>
        <div className="flex gap-1">
          <Button variant="ghost" onClick={exportConversation} title="Export">
            Export
          </Button>
          <Button variant="ghost" onClick={clearMessages} title="Clear">
            <Trash2 size={14} />
          </Button>
        </div>
      </div>
      <div className="flex flex-wrap gap-1 pb-2">
        {suggestedQuestions.map((q) => (
          <button
            key={q}
            className="rounded border border-line px-2 py-1 text-[11px] text-muted hover:border-accent hover:text-fg"
            onClick={() => send(q)}
            type="button"
          >
            {q}
          </button>
        ))}
      </div>
      <div className="flex-1 space-y-3 overflow-y-auto rounded-md border border-line bg-canvas p-3">
        {session.messages.length === 0 ? (
          <p className="text-sm text-muted">Ask why a prediction was made, which regions mattered, or what literature is available. I will not invent findings or papers.</p>
        ) : null}
        {session.messages.map((m) => (
          <article key={m.id} className={m.role === "doctor" ? "ml-8 rounded-md bg-panel p-3" : "mr-8 rounded-md border border-line bg-panel p-3"}>
            <div className="mb-1 flex items-center justify-between text-[10px] uppercase tracking-wide text-muted">
              <span>{m.role === "doctor" ? session.name || "Doctor" : "FedXAI-Med"}</span>
              <span className="font-mono">{new Date(m.createdAt).toLocaleTimeString()}</span>
            </div>
            <div className="whitespace-pre-wrap text-sm leading-relaxed">{m.content}</div>
            {m.role === "assistant" ? (
              <div className="mt-2 flex gap-1">
                <button type="button" className="p-1 text-muted hover:text-fg" title="Copy" onClick={() => navigator.clipboard.writeText(m.content)}>
                  <Copy size={13} />
                </button>
                <button type="button" className="p-1 text-muted hover:text-fg" title="Regenerate" onClick={() => send(session.messages[session.messages.indexOf(m) - 1]?.content ?? "")}>
                  <RotateCcw size={13} />
                </button>
                <button type="button" className="p-1 text-muted hover:text-fg" title="Helpful">
                  <ThumbsUp size={13} />
                </button>
                <button type="button" className="p-1 text-muted hover:text-fg" title="Not helpful">
                  <ThumbsDown size={13} />
                </button>
              </div>
            ) : null}
          </article>
        ))}
        {streaming ? <p className="text-xs text-accent">Generating grounded reply…</p> : null}
        <div ref={end} />
      </div>
      {lastCitations.length ? (
        <div className="mt-2">
          <button type="button" className="text-[11px] text-accent" onClick={() => setEvidenceOpen((v) => !v)}>
            {evidenceOpen ? "Hide" : "Show"} evidence
          </button>
          {evidenceOpen ? <EvidencePanel items={lastCitations} /> : null}
        </div>
      ) : null}
      <form
        className="mt-2 flex gap-2"
        onSubmit={(e) => {
          e.preventDefault();
          void send(draft);
        }}
      >
        <input
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          placeholder="Ask about the model output, attribution, or literature…"
          className="flex-1 rounded-md border border-line bg-input px-3 py-2 text-sm"
        />
        <Button type="submit" disabled={streaming}>
          Send
        </Button>
      </form>
    </div>
  );
}
