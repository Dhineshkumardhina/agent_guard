import React, { useEffect, useState } from "react";
import { Shield, ExternalLink, Terminal } from "lucide-react";
import { getHealth } from "../../api";
import { Badge } from "../common/Badges";

export const Header: React.FC = () => {
  const [isLive, setIsLive] = useState<boolean | null>(null);

  useEffect(() => {
    getHealth()
      .then((h) => setIsLive(h.status === "ok"))
      .catch(() => setIsLive(false));
    const timer = setInterval(() => {
      getHealth()
        .then((h) => setIsLive(h.status === "ok"))
        .catch(() => setIsLive(false));
    }, 15000);
    return () => clearInterval(timer);
  }, []);

  return (
    <header className="flex h-14 w-full items-center justify-between border-b border-[#242b3d] bg-[#0c0e14] px-6">
      {/* Brand & Subtitle */}
      <div className="flex items-center gap-3">
        <div className="flex h-8 w-8 items-center justify-center rounded border border-[#333d56] bg-[#141721] text-[#3b82f6]">
          <Shield className="h-4 w-4" />
        </div>
        <div>
          <div className="flex items-center gap-2">
            <span className="font-mono text-sm font-bold tracking-tight text-[#f0f3fa]">
              AgentGuard
            </span>
            <Badge variant="outline" size="sm">
              RESEARCH CONSOLE
            </Badge>
          </div>
        </div>
      </div>

      {/* Status & Links */}
      <div className="flex items-center gap-4 text-xs">
        {/* Backend Connectivity Status */}
        <div className="flex items-center gap-2 rounded border border-[#242b3d] bg-[#141721] px-2.5 py-1 text-[#9aa4bf]">
          <span
            className={`h-2 w-2 rounded-full ${
              isLive === true
                ? "bg-[#10b981] animate-pulse"
                : isLive === false
                ? "bg-[#ef4444]"
                : "bg-[#fbbf24]"
            }`}
          />
          <span className="font-mono text-[11px]">
            API: {isLive === true ? "CONNECTED" : isLive === false ? "OFFLINE" : "CHECKING"}
          </span>
        </div>

        {/* API Docs Link */}
        <a
          href="http://127.0.0.1:8000/docs"
          target="_blank"
          rel="noreferrer"
          className="flex items-center gap-1.5 rounded border border-[#242b3d] bg-[#141721] px-2.5 py-1 text-[#9aa4bf] transition-colors hover:border-[#3b82f6] hover:text-[#f0f3fa]"
          title="Open FastAPI Swagger Interactive Docs"
        >
          <Terminal className="h-3.5 w-3.5 text-[#5e6984]" />
          <span>OpenAPI Docs</span>
          <ExternalLink className="h-3 w-3 text-[#5e6984]" />
        </a>
      </div>
    </header>
  );
};
