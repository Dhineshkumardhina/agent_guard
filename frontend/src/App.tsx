import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import AppLayout from './components/layout/AppLayout';
import OverviewPage from './pages/OverviewPage';
import ModelComparisonPage from './pages/ModelComparisonPage';
import ExperimentDetailPage from './pages/ExperimentDetailPage';
import RunsListPage from './pages/RunsListPage';
import RunDetailPage from './pages/RunDetailPage';
import EarlyWarningPage from './pages/EarlyWarningPage';
import AblationsPage from './pages/AblationsPage';
import GeneralizationPage from './pages/GeneralizationPage';
import ExplainabilityPage from './pages/ExplainabilityPage';
import CaseStudyDetailPage from './pages/CaseStudyDetailPage';

export const App: React.FC = () => {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<AppLayout />}>
          <Route path="/" element={<OverviewPage />} />
          <Route path="/experiments" element={<ModelComparisonPage />} />
          <Route path="/experiments/:id" element={<ExperimentDetailPage />} />
          <Route path="/runs" element={<RunsListPage />} />
          <Route path="/runs/:runId" element={<RunDetailPage />} />
          <Route path="/early-warning" element={<EarlyWarningPage />} />
          <Route path="/ablations" element={<AblationsPage />} />
          <Route path="/generalization" element={<GeneralizationPage />} />
          <Route path="/explainability" element={<ExplainabilityPage />} />
          <Route path="/explainability/:id" element={<CaseStudyDetailPage />} />
          <Route
            path="*"
            element={
              <div className="flex flex-col items-center justify-center min-h-[50vh] text-center p-8">
                <h2 className="text-xl font-bold text-slate-100 mb-2">404: Route Not Found</h2>
                <p className="text-xs text-slate-400 mb-4">
                  The requested research console path does not exist.
                </p>
                <a
                  href="/"
                  className="px-3 py-1.5 rounded bg-cyan-600 hover:bg-cyan-500 text-white text-xs font-semibold"
                >
                  Return to Research Overview
                </a>
              </div>
            }
          />
        </Route>
      </Routes>
    </BrowserRouter>
  );
};

export default App;
