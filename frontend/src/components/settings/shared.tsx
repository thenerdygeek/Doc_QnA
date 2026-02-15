import { motion } from "framer-motion";
import { AlertTriangle, Loader2 } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import type { ConfigData } from "@/types/api";

// ── Config field accessor ────────────────────────────────────────

export function field<T>(config: ConfigData | null, section: string, key: string, fallback: T): T {
  if (!config || !config[section]) return fallback;
  const v = config[section][key];
  return (v ?? fallback) as T;
}

// ── Restart badge ────────────────────────────────────────────────

export function RestartBadge() {
  return (
    <Badge variant="outline" className="ml-2 border-yellow-500/50 text-yellow-600 text-[10px] dark:text-yellow-400">
      <AlertTriangle className="mr-1 h-3 w-3" /> restart required
    </Badge>
  );
}

// ── Save button ──────────────────────────────────────────────────

export function SaveButton({ onClick, saving, saved }: { onClick: () => void; saving: boolean; saved: boolean }) {
  return (
    <Button size="sm" onClick={onClick} disabled={saving} className="mt-4">
      {saving && <Loader2 className="mr-1 h-3 w-3 animate-spin" />}
      {saved ? "Saved" : "Save"}
    </Button>
  );
}

// ── Animated SVG checkmark ───────────────────────────────────────

export function AnimatedCheckmark() {
  return (
    <motion.svg
      width="14"
      height="14"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="3"
      strokeLinecap="round"
      strokeLinejoin="round"
      className="text-green-600 dark:text-green-400"
    >
      <motion.path
        d="M5 12l5 5L20 7"
        initial={{ pathLength: 0 }}
        animate={{ pathLength: 1 }}
        transition={{ duration: 0.4, ease: "easeOut" }}
      />
    </motion.svg>
  );
}

// ── Loading dots ─────────────────────────────────────────────────

export function LoadingDots() {
  return (
    <span className="inline-flex items-center gap-0.5">
      {[0, 1, 2].map((i) => (
        <motion.span
          key={i}
          className="inline-block h-1.5 w-1.5 rounded-full bg-primary"
          animate={{ opacity: [0.3, 1, 0.3] }}
          transition={{ duration: 1, repeat: Infinity, delay: i * 0.2 }}
        />
      ))}
    </span>
  );
}
