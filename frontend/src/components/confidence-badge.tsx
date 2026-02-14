import { ShieldCheck, ShieldAlert } from "lucide-react";

interface ConfidenceBadgeProps {
  passed: boolean;
  confidence: number;
}

export function ConfidenceBadge({ passed, confidence }: ConfidenceBadgeProps) {
  const pct = Math.round(confidence * 100);

  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium ring-1 ring-inset ${
        passed
          ? "bg-emerald-50 text-emerald-700 ring-emerald-500/20 dark:bg-emerald-500/10 dark:text-emerald-400 dark:ring-emerald-500/20"
          : "bg-amber-50 text-amber-700 ring-amber-500/20 dark:bg-amber-500/10 dark:text-amber-400 dark:ring-amber-500/20"
      }`}
    >
      {passed ? (
        <ShieldCheck className="h-3.5 w-3.5" />
      ) : (
        <ShieldAlert className="h-3.5 w-3.5" />
      )}
      {passed ? "Verified" : "Unverified"} ({pct}%)
    </span>
  );
}
