import { ClinicalChat } from "../components/ClinicalChat";
import { useSession } from "../context/SessionContext";

export function ChatPage() {
  const { activeCase } = useSession();
  return (
    <div className="mx-auto max-w-4xl">
      <h1 className="mb-1 text-xl font-semibold">Clinical dialogue</h1>
      <p className="mb-4 text-sm text-muted">
        Context: {activeCase ? activeCase.fileName : "no active case"}. Answers distinguish model output, attribution, retrieved evidence, and uncertainty.
      </p>
      <ClinicalChat />
    </div>
  );
}
