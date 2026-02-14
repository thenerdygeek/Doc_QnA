import { AnimatePresence, motion } from "framer-motion";
import { Loader2 } from "lucide-react";
import type { PipelineStatus } from "@/types/sse";

const STATUS_LABELS: Record<string, string> = {
  classifying: "Classifying intent\u2026",
  retrieving: "Searching documentation\u2026",
  grading: "Grading relevance\u2026",
  generating: "Generating answer\u2026",
  verifying: "Verifying answer\u2026",
};

interface StatusIndicatorProps {
  status: PipelineStatus | null;
}

export function StatusIndicator({ status }: StatusIndicatorProps) {
  if (!status || status === "complete") return null;

  const label = STATUS_LABELS[status] ?? status;

  return (
    <AnimatePresence mode="wait">
      <motion.div
        key={status}
        initial={{ opacity: 0, y: -4 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: 4 }}
        transition={{ duration: 0.2 }}
        className="inline-flex items-center gap-2 rounded-full bg-primary/8 px-3 py-1.5 text-xs font-medium text-primary"
        role="status"
        aria-live="polite"
        aria-label={label}
      >
        <Loader2 className="h-3 w-3 animate-spin" />
        <span>{label}</span>
      </motion.div>
    </AnimatePresence>
  );
}
