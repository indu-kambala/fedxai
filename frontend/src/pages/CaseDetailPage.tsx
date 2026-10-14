import { useEffect } from "react";
import { useParams } from "react-router-dom";
import { useSession } from "../context/SessionContext";
import { AnalyzePage } from "./AnalyzePage";

export function CaseDetailPage() {
  const { id } = useParams();
  const { session, setActiveCase } = useSession();
  const record = session.cases.find((c) => c.id === id);

  useEffect(() => {
    if (record && session.activeCaseId !== record.id) {
      setActiveCase(record.id);
    }
  }, [record, session.activeCaseId, setActiveCase]);

  if (!record) {
    return <p className="text-sm text-muted">Case not found in this session.</p>;
  }
  return <AnalyzePage existing={record} />;
}
