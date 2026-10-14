import { useTheme } from "../context/ThemeContext";
import { useSession } from "../context/SessionContext";
import { HOSPITALS, MODELS } from "../lib/types";
import { Card, Field, Select } from "../components/ui/primitives";

export function SettingsPage() {
  const { theme, toggle } = useTheme();
  const { session, setHospital, setModel } = useSession();

  return (
    <div className="max-w-lg space-y-4">
      <h1 className="text-xl font-semibold">Settings</h1>
      <Card className="space-y-4 p-4">
        <Field label="Theme">
          <Select
            value={theme}
            onChange={(e) => {
              if (e.target.value !== theme) toggle();
            }}
          >
            <option value="dark">Dark</option>
            <option value="light">Light</option>
          </Select>
        </Field>
        <Field label="Default hospital">
          <Select className="w-full" value={session.hospitalId} onChange={(e) => setHospital(e.target.value)}>
            {HOSPITALS.map((h) => (
              <option key={h.id} value={h.id}>
                {h.name}
              </option>
            ))}
          </Select>
        </Field>
        <Field label="Default model">
          <Select className="w-full" value={session.modelId} onChange={(e) => setModel(e.target.value)}>
            {MODELS.map((m) => (
              <option key={m.id} value={m.id}>
                {m.name}
              </option>
            ))}
          </Select>
        </Field>
        <p className="text-xs text-muted">API keys are never stored in the frontend. Hugging Face tokens stay on the server.</p>
      </Card>
    </div>
  );
}
