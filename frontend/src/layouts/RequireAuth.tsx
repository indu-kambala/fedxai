import { Navigate, Outlet } from "react-router-dom";
import { useSession } from "../context/SessionContext";

export function RequireAuth() {
  const { signedIn } = useSession();
  if (!signedIn) return <Navigate to="/login" replace />;
  return <Outlet />;
}
