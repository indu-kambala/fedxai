import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useSession } from "../context/SessionContext";
import type { Role } from "../lib/types";
import { DISCLAIMER } from "../lib/types";
import { Button, Field, Input, Select } from "../components/ui/primitives";

const ROLES: { id: Role; label: string; home: string }[] = [
  { id: "doctor", label: "Doctor", home: "/analyze" },
  { id: "researcher", label: "Researcher", home: "/experiments" },
  { id: "hospital_admin", label: "Hospital Admin", home: "/federated" },
  { id: "system_admin", label: "System Admin", home: "/dashboard" },
];

export function LoginPage() {
  const { signIn } = useSession();
  const navigate = useNavigate();
  const [name, setName] = useState("Dr. Patel");
  const [role, setRole] = useState<Role>("doctor");

  return (
    <div className="grid min-h-screen md:grid-cols-2">
      <section className="hidden flex-col justify-between border-r border-line bg-panel p-10 md:flex">
        <div>
          <p className="font-mono text-xs uppercase tracking-[0.25em] text-accent">FedXAI-Med</p>
          <h1 className="mt-4 max-w-md text-3xl font-semibold leading-tight">Federated Clinical Intelligence Platform</h1>
          <p className="mt-4 max-w-md text-sm leading-relaxed text-muted">
            Privacy-preserving federated learning, architecture-aware explanations, and grounded literature for chest X-ray decision support.
          </p>
        </div>
        <ul className="space-y-2 text-sm text-muted">
          <li>Raw patient images remain at the hospital client</li>
          <li>Central server receives model updates, not pixels</li>
          <li>Physician remains the decision-maker</li>
        </ul>
      </section>
      <section className="flex items-center justify-center p-8">
        <form
          className="w-full max-w-sm space-y-4"
          onSubmit={(e) => {
            e.preventDefault();
            signIn(name.trim() || "Clinician", role);
            navigate(ROLES.find((r) => r.id === role)?.home ?? "/dashboard");
          }}
        >
          <h2 className="text-xl font-semibold">Sign in</h2>
          <p className="text-xs leading-relaxed text-muted">{DISCLAIMER}</p>
          <Field label="Name">
            <Input value={name} onChange={(e) => setName(e.target.value)} autoComplete="name" />
          </Field>
          <Field label="Role">
            <Select className="w-full" value={role} onChange={(e) => setRole(e.target.value as Role)}>
              {ROLES.map((r) => (
                <option key={r.id} value={r.id}>
                  {r.label}
                </option>
              ))}
            </Select>
          </Field>
          <Button type="submit" className="w-full">
            Enter workstation
          </Button>
          <p className="text-[11px] text-muted">Prototype authentication. JWT/RBAC will replace this local session.</p>
        </form>
      </section>
    </div>
  );
}
