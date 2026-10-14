import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../lib/api";
import { HOSPITALS } from "../lib/types";
import { Card } from "../components/ui/primitives";
import { HospitalStatus } from "../components/HospitalStatus";
import { FederatedRoundChart } from "../components/FederatedRoundChart";

export function FederatedPage() {
  const [note, setNote] = useState("Loading…");
  const [algo, setAlgo] = useState("fedavg");

  useEffect(() => {
    api
      .federatedStatus()
      .then((s) => {
        setAlgo(s.algorithm);
        setNote(
          s.raw_images_on_wire
            ? "Configuration error: raw images must not be on the wire."
            : `Aggregator configured. Training: ${s.training}. Raw images on wire: no.`
        );
      })
      .catch(() => setNote("API offline. Five-client layout is shown; no rounds have been run."));
  }, []);

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-semibold">Federated learning</h1>
          <p className="text-sm text-muted">{note}</p>
        </div>
        <Link to="/federated/rounds" className="text-sm text-accent">
          Round log
        </Link>
      </div>
      <div className="grid gap-4 lg:grid-cols-2">
        <Card className="p-4">
          <h2 className="mb-2 text-sm font-semibold">Client status</h2>
          {HOSPITALS.map((h) => (
            <HospitalStatus key={h.id} name={h.name} status="Waiting" />
          ))}
        </Card>
        <Card className="p-4">
          <h2 className="mb-2 text-sm font-semibold">Global F1 by round</h2>
          <FederatedRoundChart data={[]} />
        </Card>
      </div>
      <Card className="p-4">
        <h2 className="mb-2 text-sm font-semibold">Algorithm comparison</h2>
        <table className="w-full text-sm">
          <thead className="text-left text-xs uppercase text-muted">
            <tr>
              <th className="py-1">Method</th>
              <th>Accuracy</th>
              <th>Macro-F1</th>
              <th>ROC-AUC</th>
            </tr>
          </thead>
          <tbody>
            {["Centralized", "FedAvg", "FedProx", "FedProx + DP"].map((name) => (
              <tr key={name} className="border-t border-line">
                <td className="py-2">{name === "FedAvg" ? `${name} (default: ${algo})` : name}</td>
                <td className="font-mono text-xs text-muted">Not evaluated</td>
                <td className="font-mono text-xs text-muted">Not evaluated</td>
                <td className="font-mono text-xs text-muted">Not evaluated</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>
    </div>
  );
}
