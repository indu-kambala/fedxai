import { Navigate, Route, Routes } from "react-router-dom";
import { AppShell } from "./layouts/AppShell";
import { RequireAuth } from "./layouts/RequireAuth";
import { LoginPage } from "./pages/LoginPage";
import { DashboardPage } from "./pages/DashboardPage";
import { AnalyzePage } from "./pages/AnalyzePage";
import { CasesPage } from "./pages/CasesPage";
import { CaseDetailPage } from "./pages/CaseDetailPage";
import { ChatPage } from "./pages/ChatPage";
import { ModelsPage } from "./pages/ModelsPage";
import { ExperimentsPage } from "./pages/ExperimentsPage";
import { FederatedPage } from "./pages/FederatedPage";
import { FederatedRoundsPage } from "./pages/FederatedRoundsPage";
import { ExplainabilityPage } from "./pages/ExplainabilityPage";
import { PrivacyPage } from "./pages/PrivacyPage";
import { KnowledgePage } from "./pages/KnowledgePage";
import { ReportsPage } from "./pages/ReportsPage";
import { AuditPage } from "./pages/AuditPage";
import { SettingsPage } from "./pages/SettingsPage";

export default function App() {
  return (
    <Routes>
      <Route path="/login" element={<LoginPage />} />
      <Route element={<RequireAuth />}>
        <Route element={<AppShell />}>
          <Route path="/" element={<Navigate to="/dashboard" replace />} />
          <Route path="/dashboard" element={<DashboardPage />} />
          <Route path="/analyze" element={<AnalyzePage />} />
          <Route path="/cases" element={<CasesPage />} />
          <Route path="/cases/:id" element={<CaseDetailPage />} />
          <Route path="/chat" element={<ChatPage />} />
          <Route path="/models" element={<ModelsPage />} />
          <Route path="/experiments" element={<ExperimentsPage />} />
          <Route path="/federated" element={<FederatedPage />} />
          <Route path="/federated/rounds" element={<FederatedRoundsPage />} />
          <Route path="/explainability" element={<ExplainabilityPage />} />
          <Route path="/privacy" element={<PrivacyPage />} />
          <Route path="/knowledge" element={<KnowledgePage />} />
          <Route path="/reports" element={<ReportsPage />} />
          <Route path="/audit" element={<AuditPage />} />
          <Route path="/settings" element={<SettingsPage />} />
        </Route>
      </Route>
      <Route path="*" element={<Navigate to="/dashboard" replace />} />
    </Routes>
  );
}
