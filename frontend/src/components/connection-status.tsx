import { useEffect, useState } from "react";
import { api } from "@/api/client";
import type { HealthResponse } from "@/types/api";

type Status = "connected" | "degraded" | "disconnected" | "checking";

interface HealthDetails {
  indexOk: boolean;
  indexChunks: number;
}

export function ConnectionStatus() {
  const [status, setStatus] = useState<Status>("checking");
  const [details, setDetails] = useState<HealthDetails | null>(null);
  const [showTooltip, setShowTooltip] = useState(false);

  useEffect(() => {
    let mounted = true;

    async function check() {
      try {
        const data: HealthResponse = await api.health();
        if (!mounted) return;

        const idx = data.components?.index;
        const indexOk = idx?.ok ?? false;
        const indexChunks = idx?.chunks ?? 0;

        setDetails({ indexOk, indexChunks });

        if (indexOk && indexChunks > 0) {
          setStatus("connected");
        } else {
          setStatus("degraded");
        }
      } catch {
        if (mounted) {
          setStatus("disconnected");
          setDetails(null);
        }
      }
    }

    check();
    const interval = setInterval(check, 30_000);
    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, []);

  if (status === "checking") return null;

  const isOk = status === "connected";
  const isDegraded = status === "degraded";

  const dotColor = isOk
    ? "bg-emerald-500"
    : isDegraded
      ? "bg-amber-500"
      : "bg-destructive";

  const pingColor = isOk
    ? "bg-emerald-400"
    : isDegraded
      ? "bg-amber-400"
      : "";

  const textColor = isOk
    ? "text-muted-foreground"
    : isDegraded
      ? "text-amber-600 dark:text-amber-400"
      : "text-destructive";

  const label = isOk ? "Connected" : isDegraded ? "Degraded" : "Offline";

  const tooltipText = status === "disconnected"
    ? "Backend: Offline"
    : `Backend: Connected | Index: ${
        details && details.indexChunks > 0
          ? `${details.indexChunks.toLocaleString()} chunks`
          : "Empty"
      }`;

  return (
    <div
      className="relative flex items-center gap-1.5"
      role="status"
      aria-label={`Backend ${status}`}
      onMouseEnter={() => setShowTooltip(true)}
      onMouseLeave={() => setShowTooltip(false)}
    >
      <span className="relative flex h-2 w-2">
        {(isOk || isDegraded) && (
          <span
            className={`absolute inline-flex h-full w-full animate-ping rounded-full ${pingColor} opacity-75`}
          />
        )}
        <span
          className={`relative inline-flex h-2 w-2 rounded-full ${dotColor}`}
        />
      </span>
      <span className={`hidden text-xs sm:inline ${textColor}`}>
        {label}
      </span>

      {/* Tooltip */}
      {showTooltip && (
        <div
          className="absolute top-full left-1/2 z-50 mt-2 -translate-x-1/2 whitespace-nowrap rounded-md border border-border bg-popover px-3 py-1.5 text-xs text-popover-foreground shadow-md"
          role="tooltip"
        >
          {tooltipText}
        </div>
      )}
    </div>
  );
}
