import { Link } from "react-router-dom";
import { useSession } from "../context/SessionContext";
import { Card } from "../components/ui/primitives";

export function CasesPage() {
  const { session, setActiveCase } = useSession();
  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-semibold">Previous cases</h1>
        <Link to="/analyze" className="text-sm text-accent">
          New case
        </Link>
      </div>
      {session.cases.length === 0 ? (
        <p className="text-sm text-muted">No local cases in this session. Images are not stored on the federated server.</p>
      ) : (
        <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
          {session.cases.map((c) => (
            <Card key={c.id} className="overflow-hidden">
              <img src={c.imageDataUrl} alt="" className="h-40 w-full object-cover bg-black" />
              <div className="p-3">
                <p className="truncate text-sm font-medium">{c.fileName}</p>
                <p className="font-mono text-[11px] text-muted">{c.id.slice(0, 8)}</p>
                <p className="mt-1 text-xs text-muted">{c.predictedClass ?? "Prediction pending"}</p>
                <Link to={`/cases/${c.id}`} className="text-sm text-accent" onClick={() => setActiveCase(c.id)}>
                  Open
                </Link>
              </div>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}
