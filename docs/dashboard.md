# Research Dashboard Documentation

## Overview

The AgentGuard Research Dashboard (`frontend/`) is an interactive Single Page Application (SPA) built with React 18, TypeScript 5, and Vite. It provides researchers and operators with an intuitive interface for inspecting multi-agent trajectories, exploring dynamic interaction graphs, comparing model benchmark evaluations, analyzing ablation studies, and visualizing multi-level explainability attributions.

---

## 1. Frontend Technology Stack

* **Framework:** React 18 with TypeScript 5 (Strict type checking enabled)
* **Build System:** Vite 5
* **Routing:** Declarative client-side routing with clean layout wrappers
* **Styling:** Custom CSS design system with CSS custom properties, responsive grids, and dark theme support
* **Testing:** Vitest 5 with React Testing Library (`jsdom` environment)
* **Linting:** Oxlint for ultra-fast static analysis

---

## 2. Dashboard Pages and Views

The application provides ten dedicated research views (`frontend/src/pages/`):

### 1. `OverviewPage.tsx`
* **Purpose:** Executive research summary and system health status.
* **Components:** Top-level KPI metric cards (Total Trajectories, Monitored Events, Evaluated Models, Mean Early Warning Lead Time), active simulation run feed, model performance overview.

### 2. `RunsListPage.tsx`
* **Purpose:** Paginated, searchable index of all executed simulation trajectories.
* **Features:** Filtering by task category (`research`, `coding`, `analysis`, `planning`), topology (`pipeline`, `star`, `mesh`, `custom`), agent count, and cascading failure status. Displays run duration and step count.

### 3. `RunDetailPage.tsx`
* **Purpose:** Comprehensive diagnostic inspection of an individual simulation run.
* **Components:**
  - **Agent Roster:** Cards showing agent roles, status, error counts, and average latency.
  - **Dynamic Graph Visualizer:** Interactive SVG network canvas rendering directed message pathways, node states, and edge weights.
  - **Event Timeline:** Step-by-step chronological audit stream of all interaction events, tool executions, and injected faults.

### 4. `ExperimentDetailPage.tsx`
* **Purpose:** Inspection of batch experiment configurations and metadata.
* **Features:** Displays immutable experiment JSON parameters, dataset splits, random seeds, and aggregated run outcomes.

### 5. `ModelComparisonPage.tsx`
* **Purpose:** Comprehensive benchmark evaluation across all 9 model families.
* **Features:** Horizon selector ($K \in \{1, 3, 5, 10, 20\}$), comparative metric tables (Precision, Recall, F1, AUROC, AUPRC, Brier Score, ECE), and interactive ROC / PR curve visualizers.

### 6. `EarlyWarningPage.tsx`
* **Purpose:** Incident-level lead time and operational hazard alerting analysis.
* **Features:** Mean and median lead time distributions, warnings emitted per trajectory, false alarm rates, and hazard threshold calibration curves ($\theta^* \in [0.10, 0.90]$).

### 7. `AblationsPage.tsx`
* **Purpose:** Interactive exploration of Phase 13 architectural and feature ablations.
* **Features:** Matrix of 9 ablation conditions vs. the reference `full_temporal_gnn`, delta-F1 rankings ($\Delta \text{F1}$), bootstrap 95% confidence intervals, and hypothesis test p-values.

### 8. `GeneralizationPage.tsx`
* **Purpose:** Exploration of Phase 14 distribution shift benchmarks.
* **Features:** Transfer evaluation across population scaling (8 and 12 agents), topology shift (Custom and Pipeline), task transfer, and unseen failure modes. Interactive In-Distribution vs. Out-of-Distribution gap visualizers.

### 9. `ExplainabilityPage.tsx`
* **Purpose:** Multi-level failure diagnosis and attribution explorer.
* **Features:** Global feature importance rankings, agent role attribution breakdowns, directed communication channel matrices, and temporal event attribution streams.

### 10. `CaseStudyDetailPage.tsx`
* **Purpose:** Deep-dive case analysis of representative failure and success archetypes.
* **Features:** Step-by-step risk trajectory chart ($P(F)$ over time), counterfactual perturbation delta comparisons, and agent risk attribution bars.

---

## 3. Frontend Quality Gates and Test Verification

The frontend maintains 100% test pass rates across four test suites (`frontend/src/test/`):
- `apiClient.test.ts`: REST API client serialization, error envelope parsing, and network retry logic (4 tests).
- `graphAndTimeline.test.tsx`: Graph rendering, node layout calculation, and timeline step sorting (2 tests).
- `commonComponents.test.tsx`: Metric cards, status badges, model comparison tables, and alert banners (5 tests).
- `routingAndPages.test.tsx`: Route navigation, view mounting, and header/sidebar layout components (1 test).

### Verification Commands
```powershell
# Run Vitest unit tests
npm.cmd test -- --run

# Run TypeScript build check
npm.cmd run build

# Run Oxlint linter
npm.cmd run lint
```
