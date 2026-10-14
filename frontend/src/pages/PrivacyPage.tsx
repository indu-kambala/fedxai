import { useEffect, useState } from "react";
import { api } from "../lib/api";
import { Card } from "../components/ui/primitives";
import { PrivacyStatus } from "../components/PrivacyStatus";

export function PrivacyPage() {
  const [privacy, setPrivacy] = useState<Awaited<ReturnType<typeof api.privacy>> | null>(null);

  useEffect(() => {
    api.privacy().then(setPrivacy).catch(() => setPrivacy(null));
  }, []);

  return (
    <div className="space-y-4">
      <h1 className="text-xl font-semibold">Privacy center</h1>
      <p className="text-sm text-muted">
        The central federated server must never receive raw patient images. Differential privacy, when enabled, is a utility tradeoff — not a claim of perfect secrecy.
      </p>
      <div className="grid gap-4 lg:grid-cols-2">
        <Card className="p-4">
          <PrivacyStatus
            dpEnabled={privacy?.differential_privacy_enabled ?? false}
            epsilon={privacy?.dp_epsilon_target ?? 8}
            delta={privacy?.dp_delta ?? 1e-5}
            encrypted={window.location.protocol === "https:"}
          />
        </Card>
        <Card className="p-4 space-y-2 text-sm">
          <p>
            <span className="text-muted">Perfect privacy claimed:</span> {String(privacy?.perfect_privacy_claimed ?? false)}
          </p>
          <p>
            <span className="text-muted">Raw images centralized:</span> {String(privacy?.raw_patient_images_centralized ?? false)}
          </p>
          <p>
            <span className="text-muted">Data retention:</span> central raw images never; case metadata configurable
          </p>
          <p>
            <span className="text-muted">Audit logging:</span> enabled in architecture; persistence in a later phase
          </p>
        </Card>
      </div>
    </div>
  );
}
