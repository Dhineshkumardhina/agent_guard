import React from "react";
import { NavLink } from "react-router-dom";
import {
  LayoutDashboard,
  BarChart2,
  Activity,
  BellRing,
  GitFork,
  Globe2,
  HelpCircle,
} from "lucide-react";

interface NavItem {
  to: string;
  label: string;
  icon: React.ComponentType<{ className?: string }>;
  badge?: string;
}

const navItems: NavItem[] = [
  { to: "/", label: "Overview", icon: LayoutDashboard },
  { to: "/experiments", label: "Model Comparison", icon: BarChart2 },
  { to: "/runs", label: "Simulation Runs", icon: Activity },
  { to: "/early-warning", label: "Early Warning", icon: BellRing },
  { to: "/ablations", label: "Ablations", icon: GitFork },
  { to: "/generalization", label: "Generalization", icon: Globe2 },
  { to: "/explainability", label: "Explainability", icon: HelpCircle },
];

export const Sidebar: React.FC = () => {
  return (
    <aside className="flex w-60 flex-col border-r border-[#242b3d] bg-[#0c0e14] py-4 select-none">
      <div className="px-4 pb-3">
        <span className="text-[10px] font-bold uppercase tracking-wider text-[#5e6984]">
          Research Console
        </span>
      </div>

      <nav className="flex-1 space-y-0.5 px-2">
        {navItems.map((item) => {
          const Icon = item.icon;
          return (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.to === "/"}
              className={({ isActive }) =>
                `group flex items-center justify-between rounded-md px-3 py-2 text-xs font-medium transition-colors ${
                  isActive
                    ? "bg-[#1b2030] text-[#3b82f6] border border-[#333d56]/50 shadow-sm"
                    : "text-[#9aa4bf] hover:bg-[#141721] hover:text-[#f0f3fa]"
                }`
              }
            >
              <div className="flex items-center gap-2.5">
                <Icon className="h-4 w-4" />
                <span>{item.label}</span>
              </div>
              {item.badge && (
                <span className="font-mono text-[10px] text-[#5e6984]">
                  {item.badge}
                </span>
              )}
            </NavLink>
          );
        })}
      </nav>

      <div className="border-t border-[#242b3d] px-4 pt-3 text-[11px] text-[#5e6984]">
        <div className="font-mono text-[10px]">Dataset: generalization_v1</div>
        <div className="font-mono text-[10px] mt-0.5">Models: 9 canonical</div>
      </div>
    </aside>
  );
};
