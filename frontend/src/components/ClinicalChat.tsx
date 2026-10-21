
import { useEffect, useRef, useState } from "react";
import {
  Copy,
  RotateCcw,
  ThumbsDown,
  ThumbsUp,
  Trash2,
} from "lucide-react";
import { useSession } from "../context/SessionContext";
import { suggestedQuestions } from "../lib/chat";
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
  const [evidenceOpen, setEvidenceOpen] = useState(false);

  const end = useRef<HTMLDivElement>(null);
  const modelName =
    MODELS.find((model) => model.id === session.modelId)?.name ??
    session.modelId;

  const lastCitations =
    [...session.messages]
      .reverse()
      .find((message) => message.role === "assistant")?.citations ?? [];

  useEffect(() => {
    end.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [session.messages, streaming]);

  async function send(text: string) {
    const question = text.trim();

    if (!question || streaming) return;

    addMessage({
      id: uid(),
      role: "doctor",
      content: question,
      createdAt: new Date().toISOString(),
    });

    setDraft("");
    setStreaming(true);

    try {
  const response = await fetch("/api/chat", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      question,
      prediction: activeCase?.predictedClass ?? null,
      probability: activeCase?.modelProbability ?? null,
      explanation: null,
      top_k: 3,
    }),
  });

  if (!response.ok) {
    throw new Error(`Chat API failed: ${response.status}`);
  }

  const result = await response.json();

  const content =
    result.clinical_support ??
    "The model returned no response. Please try again.";

  addMessage({
    id: uid(),
    role: "assistant",
    content,
    createdAt: new Date().toISOString(),
    citations: [],
  });
} catch {
      addMessage({
        id: uid(),
        role: "assistant",
        content:
          "The response could not be generated. Please try again. This prototype does not replace qualified clinical review.",
        createdAt: new Date().toISOString(),
        citations: [],
      });
    } finally {
      setStreaming(false);
    }
  }

  function exportConversation() {
    const content = session.messages
      .map(
        (message) =>
          `[${message.createdAt}] ${message.role}: ${message.content}`
      )
      .join("\n\n");

    const blob = new Blob([content], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");

    link.href = url;
    link.download = `fedxai-chat-${session.activeCaseId ?? "session"}.txt`;
    link.click();

    URL.revokeObjectURL(url);
  }

  return (
    <div
      className={
        compact
          ? "flex h-full min-h-0 min-w-0 flex-1 flex-col overflow-hidden"
          : "flex h-[calc(100dvh-8rem)] min-h-0 flex-col overflow-hidden"
      }
    >
      <div className="mb-2 flex shrink-0 items-center justify-between gap-2">
        <div className="min-w-0">
          <p className="text-sm font-semibold">Doctor ↔ FedXAI-Med</p>
          <p className="text-[11px] text-muted">
            Grounded clinical decision support. Not a diagnosis.
          </p>
        </div>

        <div className="flex shrink-0 gap-1">
          <Button variant="ghost" onClick={exportConversation} title="Export">
            Export
          </Button>
          <Button variant="ghost" onClick={clearMessages} title="Clear">
            <Trash2 size={14} />
          </Button>
        </div>
      </div>

      <div className="mb-2 flex shrink-0 flex-wrap gap-1">
        {suggestedQuestions.map((question) => (
          <button
            key={question}
            className="rounded border border-line px-2 py-1 text-left text-[11px] text-muted hover:border-accent hover:text-fg"
            onClick={() => void send(question)}
            type="button"
            disabled={streaming}
          >
            {question}
          </button>
        ))}
      </div>

      <div className="min-h-0 min-w-0 flex-1 space-y-3 overflow-y-auto rounded-md border border-line bg-canvas p-3">
        {session.messages.length === 0 && (
          <p className="text-sm leading-relaxed text-muted">
            Ask about model output, explainability, or available literature.
            Predictions and explanations require qualified clinical review.
          </p>
        )}

        {session.messages.map((message) => (
          <article
            key={message.id}
            className={
              message.role === "doctor"
                ? "ml-4 rounded-md bg-panel p-3 sm:ml-8"
                : "mr-2 rounded-md border border-line bg-panel p-3 sm:mr-8"
            }
          >
            <div className="mb-1 flex flex-wrap items-center justify-between gap-1 text-[10px] uppercase tracking-wide text-muted">
              <span>
                {message.role === "doctor"
                  ? session.name || "Doctor"
                  : "FedXAI-Med"}
              </span>
              <span className="font-mono">
                {new Date(message.createdAt).toLocaleTimeString()}
              </span>
            </div>

            <div className="whitespace-pre-wrap break-words text-sm leading-relaxed">
              {message.content}
            </div>

            {message.role === "assistant" && (
              <div className="mt-2 flex gap-2">
                <button
                  type="button"
                  className="p-1 text-muted hover:text-fg"
                  title="Copy"
                  onClick={() => void navigator.clipboard.writeText(message.content)}
                >
                  <Copy size={13} />
                </button>

                <button
                  type="button"
                  className="p-1 text-muted hover:text-fg"
                  title="Regenerate"
                  onClick={() => {
                    const index = session.messages.findIndex(
                      (item) => item.id === message.id
                    );
                    const previous = session.messages[index - 1];
                    if (previous?.role === "doctor") {
                      void send(previous.content);
                    }
                  }}
                >
                  <RotateCcw size={13} />
                </button>

                <button
                  type="button"
                  className="p-1 text-muted hover:text-fg"
                  title="Helpful"
                  aria-label="Helpful"
                >
                  <ThumbsUp size={13} />
                </button>

                <button
                  type="button"
                  className="p-1 text-muted hover:text-fg"
                  title="Not helpful"
                  aria-label="Not helpful"
                >
                  <ThumbsDown size={13} />
                </button>
              </div>
            )}
          </article>
        ))}

        {streaming && (
          <p className="text-xs text-accent">Generating response...</p>
        )}

        <div ref={end} />
      </div>

      {lastCitations.length > 0 && (
        <div className="mt-2 max-h-32 shrink-0 overflow-y-auto">
          <button
            type="button"
            className="text-[11px] text-accent"
            onClick={() => setEvidenceOpen((value) => !value)}
          >
            {evidenceOpen ? "Hide" : "Show"} evidence ({lastCitations.length})
          </button>

          {evidenceOpen && <EvidencePanel items={lastCitations} />}
        </div>
      )}

      <form
        className="mt-2 flex shrink-0 items-end gap-2"
        onSubmit={(event) => {
          event.preventDefault();
          void send(draft);
        }}
      >
        <textarea
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey) {
              event.preventDefault();
              void send(draft);
            }
          }}
          placeholder="Ask about model output, attribution, or literature..."
          rows={2}
          className="min-h-[64px] min-w-0 flex-1 resize-y rounded-md border border-line bg-input px-3 py-3 text-sm leading-relaxed"
        />

        <Button type="submit" disabled={streaming || !draft.trim()}>
          Send
        </Button>
      </form>
    </div>
  );
}
