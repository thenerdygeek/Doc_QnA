import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "@/api/client";
import type { ConfigData } from "@/types/api";

export interface UseSettingsReturn {
  /** Current config (null until first fetch). */
  config: ConfigData | null;
  /** True while the initial config is loading. */
  loading: boolean;
  /** Dialog open state. */
  open: boolean;
  setOpen: (v: boolean) => void;
  /** Persist a single section. Returns restart_sections if any. */
  updateSection: (
    section: string,
    data: Record<string, unknown>,
  ) => Promise<string[]>;
  /** True while a save is in flight. */
  saving: boolean;
  /** Accumulated section names that require a restart. */
  restartRequired: string[];
}

export function useSettings(): UseSettingsReturn {
  const [open, setOpen] = useState(false);
  const [config, setConfig] = useState<ConfigData | null>(null);
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [restartRequired, setRestartRequired] = useState<string[]>([]);
  const fetched = useRef(false);

  // Lazy-fetch config on first open
  useEffect(() => {
    if (!open || fetched.current) return;
    fetched.current = true;
    setLoading(true);
    api.config
      .get()
      .then(setConfig)
      .catch((err) => console.error("Failed to load config:", err))
      .finally(() => setLoading(false));
  }, [open]);

  const updateSection = useCallback(
    async (
      section: string,
      data: Record<string, unknown>,
    ): Promise<string[]> => {
      setSaving(true);
      try {
        const res = await api.config.update({ [section]: data });
        // Refresh config from server
        const updated = await api.config.get();
        setConfig(updated);
        if (res.restart_sections.length > 0) {
          setRestartRequired((prev) => {
            const combined = new Set([...prev, ...res.restart_sections]);
            return [...combined];
          });
        }
        return res.restart_sections;
      } finally {
        setSaving(false);
      }
    },
    [],
  );

  return {
    config,
    loading,
    open,
    setOpen,
    updateSection,
    saving,
    restartRequired,
  };
}
