import { Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

export function FederatedRoundChart({ data }: { data: Array<{ round: number; f1: number | null }> }) {
  if (!data.length) {
    return (
      <div className="flex h-48 items-center justify-center text-sm text-muted">
        No federated rounds recorded. Charts stay empty until a run completes.
      </div>
    );
  }
  return (
    <div className="h-48">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data}>
          <XAxis dataKey="round" stroke="currentColor" fontSize={11} />
          <YAxis stroke="currentColor" fontSize={11} domain={[0, 1]} />
          <Tooltip />
          <Line type="monotone" dataKey="f1" stroke="#3d7ea6" dot={false} connectNulls={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
