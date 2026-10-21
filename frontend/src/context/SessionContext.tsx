
import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import type { CaseRecord, ChatMessage, Role } from "../lib/types";
import { HOSPITALS, MODELS } from "../lib/types";

type Session = {
  name: string;
  role: Role;
  hospitalId: string;
  modelId: string;
  cases: CaseRecord[];
  activeCaseId: string | null;
  messages: ChatMessage[];
};

const defaultSession: Session = {
  name: "",
  role: "doctor",
  hospitalId: HOSPITALS[0].id,
  modelId: MODELS[2].id,
  cases: [],
  activeCaseId: null,
  messages: [],
};

const SessionContext = createContext<{
  session: Session;
  signedIn: boolean;
  signIn: (name: string, role: Role) => void;
  signOut: () => void;
  setHospital: (id: string) => void;
  setModel: (id: string) => void;
  addCase: (record: CaseRecord) => void;
  updateCase: (id: string, updates: Partial<CaseRecord>) => void;
  setActiveCase: (id: string | null) => void;
  activeCase: CaseRecord | null;
  addMessage: (message: ChatMessage) => void;
  clearMessages: () => void;
} | null>(null);

function load(): Session {
  try {
    const raw = sessionStorage.getItem("fedxai-session");
    if (!raw) return defaultSession;
    return { ...defaultSession, ...JSON.parse(raw) };
  } catch {
    return defaultSession;
  }
}

export function SessionProvider({ children }: { children: ReactNode }) {
  const [session, setSession] = useState<Session>(load);

  const persist = useCallback((next: Session) => {
    sessionStorage.setItem("fedxai-session", JSON.stringify(next));
    setSession(next);
  }, []);

  const signedIn = Boolean(session.name);

  const value = useMemo(() => {
    const activeCase =
      session.cases.find((c) => c.id === session.activeCaseId) ?? null;

    return {
      session,
      signedIn,
      signIn: (name: string, role: Role) =>
        persist({ ...session, name, role }),
      signOut: () => {
        sessionStorage.removeItem("fedxai-session");
        setSession(defaultSession);
      },
      setHospital: (hospitalId: string) =>
        persist({ ...session, hospitalId }),
      setModel: (modelId: string) =>
        persist({ ...session, modelId }),
      addCase: (record: CaseRecord) =>
        persist({
          ...session,
          cases: [record, ...session.cases],
          activeCaseId: record.id,
          messages: [],
        }),
      updateCase: (id: string, updates: Partial<CaseRecord>) =>
        persist({
          ...session,
          cases: session.cases.map((record) =>
            record.id === id ? { ...record, ...updates } : record
          ),
        }),
      setActiveCase: (id: string | null) =>
        persist({ ...session, activeCaseId: id, messages: [] }),
      activeCase,
      addMessage: (message: ChatMessage) =>
        persist({
          ...session,
          messages: [...session.messages, message],
        }),
      clearMessages: () => persist({ ...session, messages: [] }),
    };
  }, [persist, session, signedIn]);

  return (
    <SessionContext.Provider value={value}>
      {children}
    </SessionContext.Provider>
  );
}

export function useSession() {
  const ctx = useContext(SessionContext);
  if (!ctx) throw new Error("SessionProvider missing");
  return ctx;
}
