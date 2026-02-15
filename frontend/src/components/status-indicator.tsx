import { motion } from "framer-motion";
import { Check } from "lucide-react";
import type { PipelineStatus } from "@/types/sse";

const PIPELINE_STEPS: { key: PipelineStatus; label: string; short: string }[] = [
  { key: "classifying", label: "Classifying", short: "Classify" },
  { key: "retrieving", label: "Searching", short: "Search" },
  { key: "grading", label: "Grading", short: "Grade" },
  { key: "generating", label: "Generating", short: "Generate" },
  { key: "verifying", label: "Verifying", short: "Verify" },
];

interface StatusIndicatorProps {
  status: PipelineStatus | null;
}

export function StatusIndicator({ status }: StatusIndicatorProps) {
  if (!status || status === "complete") return null;

  const activeIdx = PIPELINE_STEPS.findIndex((s) => s.key === status);

  return (
    <motion.div
      initial={{ opacity: 0, y: -4 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25 }}
      className="flex items-center gap-1 overflow-x-auto py-1"
      role="status"
      aria-live="polite"
      aria-label={PIPELINE_STEPS[activeIdx]?.label ?? status}
    >
      {PIPELINE_STEPS.map((step, i) => {
        const isComplete = i < activeIdx;
        const isActive = i === activeIdx;
        const isFuture = i > activeIdx;

        return (
          <div key={step.key} className="flex items-center gap-1">
            {/* Dot / check */}
            <div className="flex flex-col items-center">
              {isComplete ? (
                <motion.div
                  initial={{ scale: 0.5 }}
                  animate={{ scale: 1 }}
                  className="flex h-4 w-4 items-center justify-center rounded-full bg-emerald-500/15"
                >
                  <Check className="h-2.5 w-2.5 text-emerald-600 dark:text-emerald-400" />
                </motion.div>
              ) : isActive ? (
                <motion.div
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  transition={{ type: "spring", stiffness: 400, damping: 20 }}
                  className="step-dot-active flex h-4 w-4 items-center justify-center rounded-full bg-primary"
                >
                  <div className="h-1.5 w-1.5 rounded-full bg-primary-foreground" />
                </motion.div>
              ) : (
                <div className="flex h-4 w-4 items-center justify-center">
                  <div className="h-1.5 w-1.5 rounded-full bg-muted-foreground/30" />
                </div>
              )}
              {/* Label below dot — hidden on very narrow screens, shows only active */}
              <span
                className={`mt-0.5 text-[9px] font-medium leading-none whitespace-nowrap ${isActive
                    ? "text-primary"
                    : isComplete
                      ? "text-emerald-600 dark:text-emerald-400"
                      : "text-muted-foreground/40 hidden @[360px]:inline"
                  }`}
              >
                {step.short}
              </span>
            </div>
            {/* Connector line */}
            {i < PIPELINE_STEPS.length - 1 && (
              <div
                className={`mb-3 h-px w-3 shrink-0 sm:w-5 ${isFuture ? "bg-muted-foreground/15" : "bg-primary/30"
                  }`}
              />
            )}
          </div>
        );
      })}
    </motion.div>
  );
}
