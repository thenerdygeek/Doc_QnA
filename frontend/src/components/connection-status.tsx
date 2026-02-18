import { useEffect, useState } from "react";
import { api } from "@/api/client";
import type { HealthResponse } from "@/types/api";

type Status = "connected" | "degraded" | "disconnected" | "checking";

interface HealthDetails {
  indexOk: boolean;
  indexChunks: number;
  embeddingModel?: string;
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
        const embeddingModel = data.embedding_model?.resolved;

        setDetails({ indexOk, indexChunks, embeddingModel });

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
      : "bg-amber-600";

  const pingColor = isOk
    ? "bg-emerald-400"
    : isDegraded
      ? "bg-amber-400"
      : "";

  const textColor = isOk
    ? "text-muted-foreground"
    : isDegraded
      ? "text-amber-600 dark:text-amber-400"
      : "text-amber-700 dark:text-amber-400";

  // Soften "Offline" → "Not connected" (less alarming for initial setup)
  const label = isOk ? "Connected" : isDegraded ? "Degraded" : "Not connected";

  // Derive short model label: "nomic-ai/nomic-embed-text-v1.5" → "nomic-embed-text-v1.5"
  const modelLabel = details?.embeddingModel?.split("/").pop() ?? "";

  const tooltipText = status === "disconnected"
    ? "Backend: Not connected — start the server to begin"
    : `Backend: Connected | Index: ${details && details.indexChunks > 0
      ? `${details.indexChunks.toLocaleString()} chunks`
      : "Empty"
    }${modelLabel ? ` | Model: ${modelLabel}` : ""}`;

  return (
    <div
      className="relative flex items-center gap-1.5"
      role="status"
      aria-label={`Backend ${status}`}
      onMouseEnter={() => setShowTooltip(true)}
      onMouseLeave={() => setShowTooltip(false)}
      onFocus={() => setShowTooltip(true)}
      onBlur={() => setShowTooltip(false)}
      tabIndex={0}
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
